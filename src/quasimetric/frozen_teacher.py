"""A frozen quasimetric model used only as a regression target: a teacher checkpoint's encoder and
QRL head, held in eval mode with no gradient and no optimizer group. Its output is the (B,N,N)
within-clip distance matrix the spectral term's quasimetric kernel is built from. The unused
halves of the teacher (predictor, action encoder, prediction heads) are replaced with Identity
after loading so they are not carried in memory or written into every checkpoint."""
import torch
import torch.nn as nn

from losses import QRLHead

_UNUSED = ("predictor", "action_encoder", "pred_proj", "pred_norm")


class FrozenQRLTeacher(nn.Module):
    """Args:
        ckpt:       teacher checkpoint. Accepts a step snapshot ({"model", "heads"}), a Lightning
                    checkpoint ({"state_dict"} with model./quasimetric_head. prefixes), or a
                    weights_epoch_*.pt paired with its heads_epoch_*.pt sidecar.
        model_cfg:  hydra config for the teacher's world model (loaded strictly).
        head_cfg:   kwargs for the teacher's QRLHead.
        symmetrize: "none" (the directed IQE quasimetric d(i->j)) or "mean" (0.5 * (d(i,j) + d(j,i)));
                    applied to every distance this object hands out.
    """

    def __init__(self, ckpt, model_cfg, head_cfg, symmetrize="none"):
        super().__init__()
        assert symmetrize in ("mean", "none"), f"teacher symmetrize={symmetrize!r}; expected mean | none"
        self.symmetrize = str(symmetrize)
        import hydra
        from pathlib import Path

        blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        msd = blob.get("model", None)
        # the head is stored under its module name: quasimetric_head, or qrl_head for the teachers
        # trained before the rename
        heads = blob.get("heads", {}) or {}
        hsd = heads.get("quasimetric_head", heads.get("qrl_head", None))
        if msd is None and "state_dict" in blob:                      # Lightning checkpoint
            sd = blob["state_dict"]
            msd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
            for pre in ("quasimetric_head.", "qrl_head."):
                hsd = hsd or {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        if msd is None:                                               # weights_epoch_*.pt
            msd = blob
            sc = Path(str(ckpt)).with_name(
                Path(str(ckpt)).name.replace("weights_epoch_", "heads_epoch_"))
            assert sc.exists(), (
                f"{Path(str(ckpt)).name} carries no head and its sidecar {sc.name} is missing -- "
                f"a teacher without its QRL head defines no kernel")
            heads = torch.load(sc, map_location="cpu")
            hsd = heads.get("quasimetric_head", heads.get("qrl_head", None))
        assert hsd, f"no quasimetric head state found in {ckpt}"

        self.model = hydra.utils.instantiate(model_cfg)
        self.model.load_state_dict(msd, strict=True)
        self.head = QRLHead(**head_cfg)
        self.head.load_state_dict(hsd, strict=True)
        for name in _UNUSED:
            if hasattr(self.model, name):
                setattr(self.model, name, nn.Identity())

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        n = sum(p.numel() for p in self.parameters())
        print(f"==> frozen teacher: {ckpt} ({n:,} params, all frozen, symmetrize={self.symmetrize})", flush=True)

    def train(self, mode=True):
        """Stay in eval whatever the trainer does to the module tree."""
        return super().train(False)

    def _sym(self, d):
        """Apply the configured symmetrisation to a distance block whose last two dims are the
        pair axes; works for (M,M) and (B,N,N) alike."""
        if self.symmetrize == "none":
            return d
        return 0.5 * (d + d.transpose(-1, -2))

    @torch.no_grad()
    def z_of(self, pixels):
        """(B,N,C,H,W) normalized pixels -> (B,N,z), the teacher's IQE latents."""
        info = self.model.encode({"pixels": pixels})
        return self.head.encode_z(info["emb"].float())

    @torch.no_grad()
    def key_dist_from_z(self, z):
        """(B,N,z) -> (B,N,N) symmetrised teacher distances."""
        return self._sym(self.head.pairwise_dist(z))

    @torch.no_grad()
    def all_pairs_dist_from_z(self, z):
        """(M,z) or (1,M,z) latents -> (M,M): key_dist_from_z over one flat block of frames."""
        if z.dim() == 2:
            z = z.unsqueeze(0)
        assert z.shape[0] == 1, (
            f"all_pairs_dist_from_z takes one flat block of frames, got {z.shape[0]} clips")
        return self.key_dist_from_z(z)[0]

    @torch.no_grad()
    def all_pairs_dist(self, pixels):
        """(B,N,C,H,W) normalized pixels -> (M,M) over the whole batch, M = B*N."""
        z = self.z_of(pixels)
        return self.all_pairs_dist_from_z(z.reshape(1, -1, z.shape[-1]))
