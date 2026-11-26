# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""k-NN, L-BFGS logistic regression, and MLP probes for frozen embeddings."""

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn

from .._errors import DependencyNotFoundError
from .._utils import lazy_import, resolve_device
from .linear import _inner_splits, _resolve_outer_splits, _valid_feature_label_mask
from .metrics import confusion_matrix

DEFAULT_LOGREG_CS: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4)


def knn_probe_score(
    features: NDArray[np.floating[Any]],
    labels: NDArray[Any],
    *,
    k: int = 5,
    folds: int = 5,
    seed: int = 0,
    device: str | torch.device = 'cpu',
    test_mask: NDArray[np.bool_] | None = None,
    fold_assign: NDArray[np.integer[Any]] | None = None,
    confusions: list[NDArray[np.int64]] | None = None,
) -> tuple[float, list[float]]:
    """Evaluate classification accuracy with a k-NN probe on standardized features.

    Uses ``faissknn`` via :func:`neuralftw._utils.lazy_import` when installed and
    falls back cleanly to :class:`sklearn.neighbors.KNeighborsClassifier`.

    Args:
        features: Two-dimensional feature matrix of shape ``(N, D)``.
        labels: One-dimensional class label array of shape ``(N,)``.
        k: Number of nearest neighbors (must be at least 1).
        folds: Number of CV folds when neither *test_mask* nor *fold_assign* is set.
        seed: Random seed for fold partitioning.
        device: Target device string or object (used when ``faissknn`` is available).
        test_mask: Optional boolean mask marking held-out test rows.
        fold_assign: Optional integer fold-group assignment per row.
        confusions: Optional list populated with one ``[C, C]`` confusion matrix per
            scored outer fold.

    Returns:
        Tuple ``(mean_accuracy, fold_accuracies)``.

    Raises:
        ValueError: If *k* < 1, fewer than 2 valid samples exist, or splits are invalid.
    """
    if k < 1:
        raise ValueError(f'k must be at least 1, got {k}.')
    dev = resolve_device(device)
    valid = _valid_feature_label_mask(features, labels, 'classification')
    n_valid = int(valid.sum())
    if n_valid < 2:
        raise ValueError(f'Need at least 2 valid samples, got {n_valid}.')

    feats = np.asarray(features[valid], dtype=np.float32)
    classes, y = np.unique(np.asarray(labels)[valid], return_inverse=True)
    n_classes = len(classes)

    outer_splits, _ = _resolve_outer_splits(
        n_valid, valid, folds, seed, test_mask, fold_assign
    )

    try:
        faissknn_mod = lazy_import('faissknn', extra='knn')
    except DependencyNotFoundError:
        faissknn_mod = None

    faiss_dev = 'cuda' if dev.type == 'cuda' else 'cpu'
    fold_scores: list[float] = []
    for tr_idx, te_idx in outer_splits:
        scaler = StandardScaler().fit(feats[tr_idx])
        x_tr = scaler.transform(feats[tr_idx]).astype(np.float32)
        x_te = scaler.transform(feats[te_idx]).astype(np.float32)
        y_tr = y[tr_idx]
        y_te = y[te_idx]
        k_eff = min(k, len(tr_idx))
        if faissknn_mod is not None:
            model = faissknn_mod.FaissKNNClassifier(
                n_neighbors=k_eff, n_classes=n_classes, device=faiss_dev, metric='l2'
            )
            model.fit(x_tr, y_tr)
            pred = np.asarray(model.predict(x_te), dtype=np.int64).ravel()
        else:
            clf = KNeighborsClassifier(n_neighbors=k_eff, metric='euclidean')
            clf.fit(x_tr, y_tr)
            pred = np.asarray(clf.predict(x_te), dtype=np.int64).ravel()
        if confusions is not None:
            confusions.append(confusion_matrix(y_te, pred, n_classes))
        fold_scores.append(float(np.mean(pred == y_te)))

    return float(np.mean(fold_scores)), fold_scores


class LBFGSLogReg:
    """Single-label multinomial logistic regression solved with PyTorch L-BFGS.

    Optimizes the scikit-learn matched objective
    ``mean(CrossEntropy) + 0.5 / (c * n) * ||W||_F^2`` with an unpenalized bias.
    """

    def __init__(
        self,
        c: float = 1.0,
        max_iter: int = 2000,
        tol: float = 1e-6,
        device: str | torch.device = 'cpu',
        seed: int = 0,
    ) -> None:
        """Initialize the L-BFGS logistic regression estimator.

        Args:
            c: Inverse L2 regularization strength (must be positive).
            max_iter: Maximum L-BFGS iterations (must be at least 1).
            tol: Convergence tolerance on function value change (must be positive).
            device: Target PyTorch device string or object.
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If *c*, *max_iter*, *tol*, or *device* are invalid.
        """
        if not np.isfinite(c) or c <= 0.0:
            raise ValueError(f'c must be finite and positive, got {c}.')
        if max_iter < 1:
            raise ValueError(f'max_iter must be at least 1, got {max_iter}.')
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError(f'tol must be finite and positive, got {tol}.')
        self.c = float(c)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.device = resolve_device(device)
        self.seed = int(seed)
        self.model: nn.Linear | None = None

    def fit(self, x: Tensor, y: Tensor, n_classes: int | None = None) -> 'LBFGSLogReg':
        """Fit logistic regression weights on standardized features.

        Args:
            x: Training feature tensor of shape ``(N, D)``.
            y: Training class indices tensor of shape ``(N,)`` with integer dtype.
            n_classes: Optional explicit number of classes (defaults to ``max(y) + 1``).

        Returns:
            The fitted :class:`LBFGSLogReg` instance.
        """
        torch.manual_seed(self.seed)
        x_t = x.to(self.device, dtype=torch.float32).contiguous()
        y_t = y.to(self.device, dtype=torch.long).contiguous()
        n, d = x_t.shape
        num_classes = int(y_t.max().item()) + 1 if n_classes is None else int(n_classes)
        model = nn.Linear(d, num_classes, bias=True).to(self.device)
        nn.init.zeros_(model.weight)
        nn.init.zeros_(model.bias)
        crit = nn.CrossEntropyLoss(reduction='mean')
        reg = 0.5 / (self.c * float(n))
        opt = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=self.max_iter,
            history_size=10,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-7,
            tolerance_change=self.tol * 0.1,
        )

        def closure() -> Tensor:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x_t), y_t) + reg * model.weight.mul(model.weight).sum()
            loss.backward()
            return loss

        opt.step(closure)
        model.eval()
        self.model = model
        return self

    @torch.no_grad()
    def predict(self, x: Tensor) -> NDArray[np.int64]:
        """Predict class indices for input features.

        Args:
            x: Feature tensor of shape ``(N, D)``.

        Returns:
            Predicted integer class indices with shape ``(N,)`` and dtype ``int64``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.model is None:
            raise RuntimeError('LBFGSLogReg must be fitted before calling predict.')
        x_t = x.to(self.device, dtype=torch.float32).contiguous()
        return self.model(x_t).argmax(dim=1).cpu().numpy().astype(np.int64)


def logreg_probe_score(
    features: NDArray[np.floating[Any]],
    labels: NDArray[Any],
    *,
    folds: int = 5,
    seed: int = 0,
    device: str | torch.device = 'cpu',
    c_values: tuple[float, ...] = DEFAULT_LOGREG_CS,
    max_iter: int = 500,
    test_mask: NDArray[np.bool_] | None = None,
    fold_assign: NDArray[np.integer[Any]] | None = None,
    confusions: list[NDArray[np.int64]] | None = None,
) -> tuple[float, list[float]]:
    """Evaluate classification accuracy with nested-CV L-BFGS logistic regression.

    Selects the regularization parameter ``C`` strictly inside each outer training
    pool via nested inner CV (leave-one-group-out when ``fold_assign`` has at least
    3 groups, otherwise random inner folds), avoiding test-fold leakage.

    Args:
        features: Two-dimensional feature matrix of shape ``(N, D)``.
        labels: One-dimensional class label array of shape ``(N,)``.
        folds: Number of CV folds.
        seed: Random seed for fold partitioning and optimizer seeding.
        device: Target PyTorch device string or object.
        c_values: Candidate positive ``C`` regularization values.
        max_iter: Maximum L-BFGS iterations per fit.
        test_mask: Optional boolean mask marking held-out test rows.
        fold_assign: Optional integer fold-group assignment per row.
        confusions: Optional list populated with per-fold confusion matrices.

    Returns:
        Tuple ``(mean_accuracy, fold_accuracies)``.

    Raises:
        ValueError: If *c_values* is empty, fewer than 2 valid samples exist, or splits
            are invalid.
    """
    if len(c_values) == 0:
        raise ValueError('c_values must be non-empty.')
    dev = resolve_device(device)
    valid = _valid_feature_label_mask(features, labels, 'classification')
    n_valid = int(valid.sum())
    if n_valid < 2:
        raise ValueError(f'Need at least 2 valid samples, got {n_valid}.')

    feats = np.asarray(features[valid], dtype=np.float32)
    classes, y = np.unique(np.asarray(labels)[valid], return_inverse=True)
    n_classes = len(classes)

    outer_splits, outer_groups = _resolve_outer_splits(
        n_valid, valid, folds, seed, test_mask, fold_assign
    )

    fold_scores: list[float] = []
    for f, (tr_idx, te_idx) in enumerate(outer_splits):
        if len(c_values) == 1:
            best_c = float(c_values[0])
        else:
            in_splits = _inner_splits(tr_idx, outer_groups, f, folds, seed)
            c_scores = np.zeros(len(c_values), dtype=np.float64)
            for in_tr, in_val in in_splits:
                in_scaler = StandardScaler().fit(feats[in_tr])
                x_itr = torch.as_tensor(
                    in_scaler.transform(feats[in_tr]), dtype=torch.float32
                )
                x_ival = torch.as_tensor(
                    in_scaler.transform(feats[in_val]), dtype=torch.float32
                )
                y_itr = torch.as_tensor(y[in_tr], dtype=torch.long)
                y_ival = y[in_val]
                for ci, c_val in enumerate(c_values):
                    clf = LBFGSLogReg(
                        c=float(c_val), max_iter=max_iter, device=dev, seed=seed
                    ).fit(x_itr, y_itr, n_classes=n_classes)
                    c_scores[ci] += float(np.mean(clf.predict(x_ival) == y_ival))
            best_c = float(c_values[int(np.argmax(c_scores))])

        scaler = StandardScaler().fit(feats[tr_idx])
        x_tr = torch.as_tensor(scaler.transform(feats[tr_idx]), dtype=torch.float32)
        x_te = torch.as_tensor(scaler.transform(feats[te_idx]), dtype=torch.float32)
        y_tr = torch.as_tensor(y[tr_idx], dtype=torch.long)
        y_te = y[te_idx]
        model = LBFGSLogReg(c=best_c, max_iter=max_iter, device=dev, seed=seed).fit(
            x_tr, y_tr, n_classes=n_classes
        )
        pred = model.predict(x_te)
        if confusions is not None:
            confusions.append(confusion_matrix(y_te, pred, n_classes))
        fold_scores.append(float(np.mean(pred == y_te)))

    return float(np.mean(fold_scores)), fold_scores


def mlp_probe_score(
    features: NDArray[np.floating[Any]],
    labels: NDArray[Any],
    *,
    task_type: str = 'regression',
    folds: int = 5,
    seed: int = 0,
    device: str | torch.device = 'cpu',
    hidden: int = 128,
    dropout: float = 0.5,
    epochs: int = 100,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 12,
    test_mask: NDArray[np.bool_] | None = None,
    fold_assign: NDArray[np.integer[Any]] | None = None,
    confusions: list[NDArray[np.int64]] | None = None,
) -> tuple[float, list[float]]:
    """Evaluate frozen features with a two-layer MLP probe and early stopping.

    Standardizes features on each training fold (zeroing channels with ``std <= 1e-4``
    and clipping z-scores to ``[-10, 10]``), filters non-finite features and labels,
    reserves a shuffled 10% validation slice for early stopping, and returns R^2
    (regression) or accuracy (classification).

    Args:
        features: Two-dimensional feature matrix of shape ``(N, D)``.
        labels: One-dimensional target array of shape ``(N,)``.
        task_type: Either ``'regression'`` or ``'classification'``.
        folds: Number of CV folds.
        seed: Random seed for fold splitting, validation shuffling, and weight init.
        device: Target PyTorch device string or object.
        hidden: Hidden layer width (must be positive).
        dropout: Dropout probability in ``[0, 1)``.
        epochs: Maximum training epochs (must be positive).
        batch_size: Mini-batch size (must be positive).
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        patience: Early-stopping patience in epochs (must be positive).
        test_mask: Optional boolean mask marking held-out test rows.
        fold_assign: Optional integer fold-group assignment per row.
        confusions: Optional list populated with per-fold confusion matrices
            (classification only).

    Returns:
        Tuple ``(mean_score, fold_scores)``.

    Raises:
        ValueError: If hyperparameters, inputs, or splits are invalid.
    """
    if hidden < 1 or epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError('hidden, epochs, batch_size, and patience must be positive.')
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f'dropout must be in [0, 1), got {dropout}.')
    dev = resolve_device(device)
    valid = _valid_feature_label_mask(features, labels, task_type)
    n_valid = int(valid.sum())
    if n_valid < 2:
        raise ValueError(f'Need at least 2 valid samples, got {n_valid}.')

    feats = np.asarray(features[valid], dtype=np.float32)
    lab = np.asarray(labels)[valid]
    is_reg = task_type == 'regression'
    if is_reg:
        y_raw = lab.astype(np.float32)
        y_cls = np.empty(0, dtype=np.int64)
        n_out = 1
    else:
        classes, y_cls = np.unique(lab, return_inverse=True)
        y_raw = np.empty(0, dtype=np.float32)
        n_out = len(classes)

    outer_splits, _ = _resolve_outer_splits(
        n_valid, valid, folds, seed, test_mask, fold_assign
    )

    fold_scores: list[float] = []
    for tr_i, te_i in outer_splits:
        torch.manual_seed(seed)
        mean = feats[tr_i].mean(axis=0)
        std = feats[tr_i].std(axis=0)
        keep = std > 1e-4
        sd = np.where(keep, std, 1.0)

        def _stdz(
            arr: NDArray[np.float32],
            m: NDArray[np.float32] = mean,
            s: NDArray[np.float32] = sd,
            k_mask: NDArray[np.bool_] = keep,
        ) -> Tensor:
            z = (arr - m) / s
            z[:, ~k_mask] = 0.0
            return torch.as_tensor(
                np.clip(z, -10.0, 10.0), dtype=torch.float32, device=dev
            )

        xtr = _stdz(feats[tr_i])
        xte = _stdz(feats[te_i])
        if is_reg:
            ymu = float(y_raw[tr_i].mean())
            ysd = float(y_raw[tr_i].std() + 1e-8)
            y_tr_t = torch.as_tensor(
                (y_raw[tr_i] - ymu) / ysd, dtype=torch.float32, device=dev
            ).unsqueeze(1)
            y_te_norm = (y_raw[te_i] - ymu) / ysd
        else:
            y_tr_t = torch.as_tensor(y_cls[tr_i], dtype=torch.long, device=dev)
            y_te_norm = np.empty(0, dtype=np.float32)

        ntr = len(tr_i)
        nval = max(1, ntr // 10)
        gen = torch.Generator(device='cpu').manual_seed(seed)
        rp = torch.randperm(ntr, generator=gen).to(dev)
        xtr_s, ytr_s = xtr[rp], y_tr_t[rp]
        xv, yv = xtr_s[:nval], ytr_s[:nval]
        xt, yt = xtr_s[nval:], ytr_s[nval:]

        net = nn.Sequential(
            nn.Linear(feats.shape[1], hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_out),
        ).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn: nn.Module = nn.MSELoss() if is_reg else nn.CrossEntropyLoss()

        best_val = float('inf')
        best_state: dict[str, Tensor] = {
            k: v.detach().clone() for k, v in net.state_dict().items()
        }
        since = 0
        for _ in range(epochs):
            net.train()
            for start in range(0, len(xt), batch_size):
                sl = slice(start, start + batch_size)
                opt.zero_grad(set_to_none=True)
                loss_fn(net(xt[sl]), yt[sl]).backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(loss_fn(net(xv), yv))
            if vl < best_val - 1e-5:
                best_val = vl
                since = 0
                best_state = {
                    k: v.detach().clone() for k, v in net.state_dict().items()
                }
            else:
                since += 1
                if since >= patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            pred = net(xte)
            if is_reg:
                p = pred.squeeze(1).cpu().numpy()
                ss_res = float(np.sum((y_te_norm - p) ** 2))
                ss_tot = float(np.sum((y_te_norm - y_te_norm.mean()) ** 2)) + 1e-12
                fold_scores.append(float(1.0 - ss_res / ss_tot))
            else:
                pred_cls = pred.argmax(dim=1).cpu().numpy().astype(np.int64)
                if confusions is not None:
                    confusions.append(confusion_matrix(y_cls[te_i], pred_cls, n_out))
                fold_scores.append(float(np.mean(pred_cls == y_cls[te_i])))

    return float(np.mean(fold_scores)), fold_scores


__all__ = (
    'DEFAULT_LOGREG_CS',
    'LBFGSLogReg',
    'knn_probe_score',
    'logreg_probe_score',
    'mlp_probe_score',
)
