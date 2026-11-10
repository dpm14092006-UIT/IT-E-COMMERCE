"""4.5.6 - TCN sequence classifier. Owner: Nguoi 4 (TCN is the CNN-based trend forecaster).

Requires torch. TCN plugs into validation.walk_forward via SequenceClassifier: X carries only the
(style, t_idx) keys and sequences are cut from the backward-looking feature panel, so the
information cutoff is the same as for the tabular models (4.3.6).
"""
import numpy as np

from ..constants import STYLE, TIME
from ..features.sequences import apply_standardizer, build_sequences, fit_standardizer


def receptive_field(kernel_size, dilations, convs_per_block=2):
    """(51) R = 1 + (k-1) * sum_l d_l, counting every convolution inside each residual block."""
    return 1 + (kernel_size - 1) * convs_per_block * sum(dilations)


def build_tcn(n_inputs, channels=32, kernel_size=3, dilations=(1, 2, 4), dropout=0.1, n_classes=3):
    """Causal, dilated residual TCN (Bai et al., 2018) with a 3-class head, (49)-(50)."""
    import torch.nn.functional as F
    from torch import nn

    class CausalConv(nn.Module):
        def __init__(self, c_in, c_out, d):
            super().__init__()
            self.pad = (kernel_size - 1) * d
            self.conv = nn.Conv1d(c_in, c_out, kernel_size, dilation=d)

        def forward(self, x):                       # [B, C, L]; left padding only = no look-ahead
            return self.conv(F.pad(x, (self.pad, 0)))

    class Block(nn.Module):
        def __init__(self, c_in, c_out, d):
            super().__init__()
            self.c1, self.c2 = CausalConv(c_in, c_out, d), CausalConv(c_out, c_out, d)
            self.drop = nn.Dropout(dropout)
            self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

        def forward(self, x):
            h = self.drop(F.relu(self.c1(x)))
            h = self.drop(F.relu(self.c2(h)))
            return F.relu(h + self.skip(x))

    class TCNClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            blocks, c = [], n_inputs
            for d in dilations:
                blocks.append(Block(c, channels, d))
                c = channels
            self.net = nn.Sequential(*blocks)
            self.head = nn.Linear(channels, n_classes)

        def forward(self, x):                       # x: [B, L, p]
            h = self.net(x.transpose(1, 2))         # h^TCN, (49)
            return self.head(h[:, :, -1])           # logits at the forecast origin, (50)

    return TCNClassifier()


class SequenceClassifier:
    """Trains a torch network with (weighted) cross-entropy (52)-(53) and chronological early stopping."""

    def __init__(self, panel, feature_cols, lookback, net_factory, learning_rate=1e-3, batch_size=64,
                 max_epochs=100, patience=10, class_weighting="none", seed=0, device="cpu",
                 val_fraction=0.15):
        self.panel, self.feature_cols, self.lookback = panel, feature_cols, lookback
        self.net_factory, self.lr, self.batch_size = net_factory, learning_rate, batch_size
        self.max_epochs, self.patience, self.class_weighting = max_epochs, patience, class_weighting
        self.seed, self.device, self.val_fraction = seed, device, val_fraction

    def _inputs(self, X):
        seq, mask = build_sequences(self.panel, self.feature_cols, X[[STYLE, TIME]], self.lookback)
        return seq, mask

    def _tensor(self, seq, mask):
        import torch
        z = np.concatenate([apply_standardizer(seq, self.mean_, self.std_), mask.astype(float)], axis=-1)
        return torch.tensor(z, dtype=torch.float32, device=self.device)

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seed)
        seq, mask = self._inputs(X)
        self.mean_, self.std_ = fit_standardizer(seq)              # training rows only
        Z, Y = self._tensor(seq, mask), torch.tensor(np.asarray(y), dtype=torch.long, device=self.device)
        order = np.argsort(X[TIME].to_numpy(), kind="stable")    # latest origins = early-stopping set
        n_val = max(1, int(len(order) * self.val_fraction))
        tr, va = order[:-n_val], order[-n_val:]
        weight = None
        if self.class_weighting == "balanced":
            counts = np.bincount(np.asarray(y)[tr], minlength=3).astype(float)
            weight = torch.tensor(np.where(counts > 0, counts.sum() / (3 * np.maximum(counts, 1)), 0.0),
                                  dtype=torch.float32, device=self.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        self.net_ = self.net_factory(Z.shape[-1]).to(self.device)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        best, best_state, wait = np.inf, None, 0
        rng = np.random.default_rng(self.seed)
        for _ in range(self.max_epochs):
            self.net_.train()
            for b in np.array_split(rng.permutation(tr), max(1, len(tr) // self.batch_size)):
                opt.zero_grad()
                loss_fn(self.net_(Z[b]), Y[b]).backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                v = float(loss_fn(self.net_(Z[va]), Y[va]))
            if v < best - 1e-6:
                best, wait = v, 0
                best_state = {k: t.detach().clone() for k, t in self.net_.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        import torch
        seq, mask = self._inputs(X)
        self.net_.eval()
        with torch.no_grad():
            return torch.softmax(self.net_(self._tensor(seq, mask)), dim=-1).cpu().numpy()


