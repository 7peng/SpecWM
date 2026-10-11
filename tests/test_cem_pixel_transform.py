"""The same uint8 frame reaches the encoder as the same tensor at train time
(`get_uint8_pixels_pipeline` + `GPUPixelNormalize`) and at eval time (`eval.img_transform`): RGB
order, ImageNet scale, the goal on the identical transform, and Resize(224) the identity at 224
(the assumption that lets resize-then-normalise equal normalise-then-resize)."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np                                                              # noqa: E402
import torch                                                                    # noqa: E402
from omegaconf import OmegaConf                                                 # noqa: E402

from eval import img_transform                                                  # noqa: E402
from utils import GPUPixelNormalize, get_uint8_pixels_pipeline                  # noqa: E402

SIZE = 224


def frame(seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(SIZE, SIZE, 3), dtype=np.uint8)


def eval_tensor(hwc):
    tf = img_transform(OmegaConf.create({"eval": {"img_size": SIZE}}))
    chw = torch.from_numpy(np.transpose(hwc[None], (0, 3, 1, 2)))
    return torch.stack([tf(x) for x in chw])[0]


def train_tensor(hwc):
    pipe = get_uint8_pixels_pipeline("pixels", "pixels", img_size=SIZE)
    chw = torch.from_numpy(hwc).permute(2, 0, 1)[None].contiguous()
    out = pipe({"pixels": chw})["pixels"]
    assert out.dtype == torch.uint8, out.dtype
    return GPUPixelNormalize().eval()(out.unsqueeze(1))[0][0, 0]


def test_train_and_eval_pipelines_agree():
    hwc = frame()
    e, g = eval_tensor(hwc).double(), train_tensor(hwc).double()
    assert e.shape == g.shape == (3, SIZE, SIZE)
    d = float((e - g).abs().max())
    assert d < 1e-5, f"eval.img_transform disagrees with the training pipeline by {d:.3g}"


def test_channel_order_and_goal_transform():
    red = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    red[..., 0] = 255
    got = eval_tensor(red)[:, 0, 0]
    want = torch.tensor([(1.0 - 0.485) / 0.229, (0.0 - 0.456) / 0.224, (0.0 - 0.406) / 0.225])
    assert torch.allclose(got, want, atol=1e-5), got.tolist()
    cfg = OmegaConf.create({"eval": {"img_size": SIZE}})
    chw = torch.from_numpy(np.transpose(frame()[None], (0, 3, 1, 2)))
    assert torch.allclose(img_transform(cfg)(chw[0]), img_transform(cfg)(chw[0]), atol=0)


def test_resize_is_the_identity_at_224():
    from torchvision.transforms import v2 as T
    x = torch.rand(3, SIZE, SIZE)
    assert torch.equal(T.Resize(size=SIZE)(x), x)
    assert T.Resize(size=SIZE)(torch.rand(3, 256, 256)).shape == torch.Size([3, SIZE, SIZE])


if __name__ == "__main__":
    test_train_and_eval_pipelines_agree()
    test_channel_order_and_goal_transform()
    test_resize_is_the_identity_at_224()
    print("test_cem_pixel_transform: OK")
