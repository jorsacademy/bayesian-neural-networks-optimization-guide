from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CausalConv1d(nn.Module):
    """1-D convolution with left padding only."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        if kernel_size < 1 or dilation < 1:
            raise ValueError("kernel_size and dilation must be positive")
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_padding, 0)))


class CausalTDNN(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.c1 = CausalConv1d(1, hidden, 3, 1)
        self.c2 = CausalConv1d(hidden, hidden, 3, 2)
        self.c3 = CausalConv1d(hidden, hidden, 3, 4)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        z = x.transpose(1, 2)
        z = F.relu(self.c1(z))
        z = F.relu(self.c2(z))
        z = F.relu(self.c3(z))
        return self.head(z[:, :, -1]).squeeze(-1)


def assert_causal_layer() -> None:
    layer = CausalConv1d(1, 2, kernel_size=3, dilation=2)
    probe = torch.randn(1, 1, 20, requires_grad=True)
    layer(probe)[0, :, 10].sum().backward()
    future_grad = probe.grad[:, :, 11:]
    if not torch.allclose(future_grad, torch.zeros_like(future_grad), atol=1e-8):
        raise AssertionError("future leakage detected")


def train_tdnn(Xtr, ytr, Xva, yva, epochs=180, batch_size=64, lr=1e-3, patience=20):
    model = CausalTDNN().to(DEVICE)
    opt, loss_fn = torch.optim.Adam(model.parameters(), lr=lr), nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=batch_size, shuffle=True)
    Xv, yv = torch.tensor(Xva, device=DEVICE), torch.tensor(yva, device=DEVICE)
    best_state, best_val, stale = None, float("inf"), 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("TDNN training produced no checkpoint")
    model.load_state_dict(best_state)
    return model
