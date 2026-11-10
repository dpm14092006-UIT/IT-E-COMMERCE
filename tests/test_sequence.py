import numpy as np
import pandas as pd
import pytest

from vstyle.features.sequences import apply_standardizer, build_sequences, fit_standardizer
from vstyle.models.sequence import receptive_field


def test_receptive_field_covers_lookback():
    assert receptive_field(3, [1, 2, 4], convs_per_block=1) == 15        # (51)
    assert receptive_field(3, [1, 2, 4]) >= 12


def test_sequences_pad_with_nan_at_style_start(panel):
    rows = pd.DataFrame({"style": ["A", "B"], "t_idx": [2, 20]})
    X, mask = build_sequences(panel, ["sales", "discount"], rows, lookback=4)
    assert X.shape == (2, 4, 2)
    assert np.isnan(X[0, 0]).all() and not mask[0, 0].any()               # t = -1 does not exist
    b = panel[(panel["style"] == "B") & panel["t_idx"].between(17, 20)]["sales"].to_numpy()
    assert np.allclose(X[1, :, 0], b, equal_nan=True)
    mean, std = fit_standardizer(X)
    assert not np.isnan(apply_standardizer(X, mean, std)).any()


def test_tcn_is_causal():
    torch = pytest.importorskip("torch")
    from vstyle.models.sequence import build_tcn
    net = build_tcn(n_inputs=3, channels=8, kernel_size=3, dilations=(1, 2)).eval()
    x = torch.randn(1, 10, 3)
    y = x.clone()
    y[:, 7:, :] += 5.0                                   # change only weeks 7..9
    with torch.no_grad():
        h1, h2 = net.net(x.transpose(1, 2)), net.net(y.transpose(1, 2))
    assert torch.allclose(h1[:, :, :7], h2[:, :, :7])    # earlier positions unaffected
    assert net(x).shape == (1, 3)
