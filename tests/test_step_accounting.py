"""With a second optimizer group (head_optim), Lightning's global_step advances twice per batch; the run must still
take trainer.max_steps batches, and batch_step must read the batch count."""
import sys
from pathlib import Path
import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_common import n_optimizers  # noqa: E402
from utils import batch_step  # noqa: E402


class TwoOpt(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False
        self.a, self.b = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
        self.batches = 0

    def training_step(self, batch, batch_idx):
        self.batches += 1
        for opt, net in zip(self.optimizers(), (self.a, self.b)):
            opt.zero_grad(); self.manual_backward(net(batch[0]).pow(2).mean()); opt.step()

    def configure_optimizers(self):
        return [torch.optim.SGD(self.a.parameters(), 0.1), torch.optim.SGD(self.b.parameters(), 0.1)]


def test_two_optimizer_run_takes_max_steps_batches():
    cfg = OmegaConf.create({"head_optim": {"modules": "crl_heads"}})
    assert n_optimizers(cfg) == 2 and n_optimizers(OmegaConf.create({})) == 1
    want = 7
    data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(64, 2)), batch_size=2)
    m = TwoOpt()
    t = pl.Trainer(max_steps=want * n_optimizers(cfg), max_epochs=10, logger=False, enable_checkpointing=False,
                   enable_progress_bar=False, enable_model_summary=False, accelerator="cpu")
    t.fit(m, data)
    assert m.batches == want, m.batches
    assert t.global_step == 2 * want and batch_step(t) == want
