"""Load a trained encoder + projector for offline probing.

The `emb` representation the probes read is the ViT pooled token through the projector, so both
pieces are needed. `window_cls` is the pooling rule every reader shares: the CLS token of the frame.
"""
import torch
import torch.nn as nn

from networks import MLP

EMB_DIM = 192          # ViT-tiny hidden size
PATCH = 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _vit():
    import stable_pretraining as spt
    return spt.backbone.utils.vit_hf(size="tiny", image_size=224, patch_size=PATCH,
                                     pretrained=False, use_mask_token=False)


def window_cls(enc, x, window=1):
    """CLS token per row. x is (n,3,H,W) frames, or (n,window,3,H,W) windows -> (n, EMB_DIM)."""
    if x.dim() == 5:
        n, k = x.shape[:2]
        h = enc(x.reshape(n * k, *x.shape[2:]), interpolate_pos_encoding=True)
        return h.last_hidden_state[:, 0].reshape(n, k, -1)[:, -1]
    return enc(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]


def _state_dict(ckpt):
    """The flat `encoder.*` / `projector.*` mapping from a training snapshot or a Lightning
    checkpoint (nested under `state_dict` with `model.` prefixes)."""
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = sd.get("model", sd)
    if any(k.startswith("model.encoder.") for k in sd):
        sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    return sd


def build_encoder(ckpt, device="cuda", window=1, causal=False):
    """-> (encoder, projector, n_views=1) in eval mode, from a training state dict."""
    assert window == 1 and not causal, "the paper checkpoints are single-frame encoders"
    sd = _state_dict(ckpt)
    enc = _vit()
    # the projector's norm differs by recipe and is not recorded anywhere else
    bn = any(k.startswith("projector.net.1.running_mean") for k in sd)
    pw = sd.get("projector.net.0.weight")
    assert pw is None or pw.shape[1] == EMB_DIM, (
        f"projector input {pw.shape[1]} is not the single-view width {EMB_DIM}")
    proj = MLP(input_dim=EMB_DIM, hidden_dim=2048, output_dim=EMB_DIM,
               norm_fn=nn.BatchNorm1d if bn else nn.LayerNorm)
    esd = {k[8:]: v for k, v in sd.items() if k.startswith("encoder.")}
    enc.load_state_dict(esd, strict=True)
    proj.load_state_dict({k[10:]: v for k, v in sd.items() if k.startswith("projector.")},
                         strict=True)
    return enc.to(device).eval(), proj.to(device).eval(), 1


def random_encoder(device="cuda", seed=0, window=1, causal=False, n_views=1):
    """The untrained-encoder floor: same architecture, never optimised."""
    assert window == 1 and not causal and n_views == 1, "the paper checkpoints are single-frame, single-view"
    torch.manual_seed(seed)
    proj = MLP(input_dim=EMB_DIM, hidden_dim=2048, output_dim=EMB_DIM, norm_fn=nn.LayerNorm)
    return _vit().to(device).eval(), proj.to(device).eval(), 1
