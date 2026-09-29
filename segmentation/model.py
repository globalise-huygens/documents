"""
Boundary model: combines predictor features into per-scan scores, plus the
length prior and alignment parameters, stored together as one JSON file.

  start   log-odds that a document (or subdocument) starts on scan i
  end     log-odds that a document ends on scan i
  shared  log-odds, given a start on scan i, that the previous document ends
          on that same scan (a letter ending halfway down the page with the
          next copy right below it) rather than on an earlier scan
  nondoc  log-odds that scan i belongs to no document (cover, blank page,
          table of contents, title page, ...)

All four are logistic regressions on the features, including those of
neighbouring scans ("name@-1" = previous scan, "name@+1" = next scan), fitted
on ground truth. Missing feature weights count as 0, so new predictors can be
added before refitting.
"""

import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

DEFAULT_MODEL_PATH = "segmentation/model.json"


def shifted(features: pd.DataFrame, name: str) -> np.ndarray | None:
    """Column `name`, or `base@k` = column `base` of scan i+k (0 outside the inventory)."""
    base, _, k = name.partition("@")
    if base not in features:
        return None
    col = features[base].to_numpy(dtype=float)
    if not k:
        return col
    k = int(k)
    out = np.zeros_like(col)
    if k > 0:
        out[:-k] = col[k:]
    else:
        out[-k:] = col[:k]
    return out


def design(features: pd.DataFrame, columns: list[str], shifts: tuple[int, ...]) -> pd.DataFrame:
    """Feature matrix with context columns: shift k adds `name@k` (scan i+k)."""
    out = {}
    for k in shifts:
        for c in columns:
            name = c if k == 0 else f"{c}@{k:+d}"
            out[name] = shifted(features, name)
    return pd.DataFrame(out)


@dataclass
class Logistic:
    bias: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)

    def logit(self, features: pd.DataFrame) -> np.ndarray:
        out = np.full(len(features), self.bias, dtype=float)
        for name, w in self.weights.items():
            col = shifted(features, name)
            if col is not None:
                out += w * col
        return out

    @classmethod
    def fit(cls, X: pd.DataFrame, y: np.ndarray, l2: float = 1.0, weights: np.ndarray | None = None, iters: int = 50) -> "Logistic":
        """Weighted, L2-regularised logistic regression by Newton's method (bias not penalised)."""
        w_obs = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
        names = list(X.columns)
        A = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
        # standardise for conditioning, then map weights back
        mu = A[:, 1:].mean(axis=0)
        sd = A[:, 1:].std(axis=0)
        sd[sd == 0] = 1
        Z = A.copy()
        Z[:, 1:] = (A[:, 1:] - mu) / sd
        beta = np.zeros(Z.shape[1])
        reg = np.full(Z.shape[1], l2)
        reg[0] = 0
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(Z @ beta)))
            g = Z.T @ (w_obs * (p - y)) + reg * beta
            H = (Z * (w_obs * p * (1 - p))[:, None]).T @ Z + np.diag(reg + 1e-9)
            step = np.linalg.solve(H, g)
            beta -= step
            if np.abs(step).max() < 1e-6:
                break
        w = beta[1:] / sd
        b = beta[0] - (w * mu).sum()
        return cls(float(b), {n: float(v) for n, v in zip(names, w)})


@dataclass
class LengthPrior:
    """
    log p(L) - log geometric(L; q): the correction that turns independent
    per-scan start logits into a semi-Markov model with the empirical length
    distribution of documents (in scans). Stored on log-spaced bins.
    """

    q: float = 0.08
    bin_edges: list[float] = field(default_factory=list)
    log_density: list[float] = field(default_factory=list)
    clip: float = 4.0

    def __call__(self, lengths: np.ndarray) -> np.ndarray:
        lengths = np.asarray(lengths, dtype=float)
        if not self.bin_edges:
            return np.zeros_like(lengths)
        k = np.clip(np.searchsorted(self.bin_edges, lengths, side="right") - 1, 0, len(self.log_density) - 1)
        lp = np.asarray(self.log_density)[k] - math.log(self.q) - (lengths - 1) * math.log(1 - self.q)
        return np.clip(lp, -self.clip, self.clip)

    @classmethod
    def fit(cls, lengths: np.ndarray, q: float, n_bins: int = 14, pseudo: float = 0.5) -> "LengthPrior":
        lengths = np.asarray(lengths, dtype=float)
        edges = np.unique(np.round(np.geomspace(1, max(lengths.max(), 2) + 1, n_bins + 1)))
        edges[0], edges[-1] = 1, max(edges[-1], 10000)
        counts, _ = np.histogram(lengths, bins=edges)
        widths = np.diff(edges)
        dens = (counts + pseudo) / (counts.sum() + pseudo * len(counts)) / widths
        return cls(q=float(q), bin_edges=edges.tolist(), log_density=np.log(dens).tolist())


@dataclass
class AlignParams:
    """Weights of the ToC alignment score (hand-set; tune on the validation set)."""

    w_start: float = 1.0  # weight of the start logit at the entry's start scan
    exact_number: float = 4.0  # observed page/folio number equals the entry's start number
    interp_base: float = 1.5  # position interpolated from nearby numbers ...
    interp_per_scan: float = 0.7  # ... minus this per scan of distance
    interp_window: int = 4  # candidates within this many scans of the interpolated position
    verso_shift: float = -1.5  # foliated volumes: starting on the verso (next scan) without a 'v' in the ToC
    date_match: float = 1.5  # header date within [-30 days, +1 year] of the entry date
    date_mismatch: float = -1.5  # header date more than 2 years away
    length_weight: float = 1.0  # |log(actual / expected span)| penalty between consecutive entries
    skip: float = -4.0  # ToC entry with a start number that is not placed
    same_start: float = -2.0  # two entries starting on the same scan (nested entries)
    max_skip: int = 8  # consecutive unplaced entries considered in one transition
    unnumbered_threshold: float = 0.5  # min score to place an entry without a start number
    max_candidates: int = 12


@dataclass
class SegmentationModel:
    start: Logistic = field(default_factory=Logistic)
    end: Logistic = field(default_factory=Logistic)
    shared: Logistic = field(default_factory=lambda: Logistic(bias=-2.0))
    nondoc: Logistic = field(default_factory=lambda: Logistic(bias=-3.0))
    length_prior: LengthPrior = field(default_factory=LengthPrior)
    align: AlignParams = field(default_factory=AlignParams)
    max_length: int = 1500  # longest document considered by the DP, in scans
    start_offset: float = 0.0  # added to the start/shared logits in the fine DP (precision/recall trade-off)
    length_prior_in_toc: bool = False  # apply the generic length prior inside placed ToC entries

    def save(self, path: str = DEFAULT_MODEL_PATH):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=1)

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "SegmentationModel":
        with open(path) as f:
            d = json.load(f)
        return cls(
            start=Logistic(**d["start"]),
            end=Logistic(**d.get("end", {})),
            shared=Logistic(**d["shared"]),
            nondoc=Logistic(**d.get("nondoc", {"bias": -3.0})),
            length_prior=LengthPrior(**d["length_prior"]),
            align=AlignParams(**d["align"]),
            max_length=d.get("max_length", 1500),
            start_offset=d.get("start_offset", 0.0),
            length_prior_in_toc=d.get("length_prior_in_toc", False),
        )
