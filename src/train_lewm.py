"""LeWM entry point: next-embedding MSE + SIGReg isotropic-Gaussian latent regularizer on a flat
ARPredictor (action embedding as AdaLN conditioning, ctx_len next-step targets per clip).

  MUJOCO_GL=egl python src/train_lewm.py --config-name=scene_lewm run_id=<run>
"""
import hydra
import torch
import torch.nn.functional as F

import train_common
from losses import SIGReg
from utils import GPUPixelNormalize


def sigreg_forward(self, batch, stage, cfg):
    """Encode -> next-embedding prediction -> loss = pred_weight * MSE + sigreg.weight * SIGReg."""
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    sw = cfg.loss.sigreg.weight
    pw = cfg.loss.get("pred_weight", 1.0)

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    clean_px, _ = self.gpu_aug(batch["pixels"])
    batch["pixels"] = clean_px

    output = self.model.encode(batch)
    emb = output["emb"]                        # (B, T, D)
    act_emb = output["act_emb"]
    sfs = act_emb.size(1) // emb.size(1)       # action tokens per frame-step
    # the clip's final frame carries a zero-padded action block; the context stays clear of it
    assert ctx_len <= emb.size(1) - 1, (
        f"history_size {ctx_len} reaches the clip's final frame ({emb.size(1)}), whose action "
        f"block is zero-padded past the episode end")
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len * sfs]
    tgt_emb = emb[:, n_preds:ctx_len + n_preds]
    pred_emb = self.model.predict(ctx_emb, ctx_act)
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    with torch.no_grad():
        d = pred_emb.size(-1)
        output["pred_cos"] = F.cosine_similarity(
            pred_emb.detach().reshape(-1, d).float(),
            tgt_emb.detach().reshape(-1, d).float(), dim=-1).mean()

    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = pw * output["pred_loss"] + sw * output["sigreg_loss"]
    train_common.log_step_metrics(self, output, emb, stage)
    return output


def build_loss_modules(cfg, gpu_norm):
    # SIGReg scores the radial spread of the embedding; a parameter-free RMS norm on the encoder
    # output removes it, so the two are refused together on this entry point.
    _sw = float(cfg.loss.sigreg.get("weight", 0.0))
    _en = cfg.get("model", {}).get("emb_norm", True)
    assert not (_sw > 0 and _en), (
        f"loss.sigreg.weight={_sw} with model.emb_norm={_en!r}: SIGReg needs the radial degree of "
        f"freedom that emb_norm removes. Set wm.emb_norm (or model.emb_norm) to false.")
    return {"sigreg": SIGReg(**cfg.loss.sigreg.kwargs), "gpu_aug": GPUPixelNormalize()}


@hydra.main(version_base=None, config_path="../config/train", config_name="scene_lewm")
def main(cfg):
    train_common.run(cfg, sigreg_forward, build_loss_modules, gpu_norm=True)


if __name__ == "__main__":
    main()
