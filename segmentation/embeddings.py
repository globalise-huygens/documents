"""
Page embeddings and kNN start scores computed elsewhere (GloBERTise embeddings
of the paragraph and marginalia text of every scan, and a kNN document-start
classifier on them; three seeds), read from EMBEDDINGS_DIR:

  embeddings/voc_paragraph_marginalia_tl1/<inv>.npz   text_embeddings, scan_ids, empty_scan_ids
  voc_start_scores_tl1/oof/<inv>.parquet              scores of the 22 validated inventories,
                                                      each by a kNN fitted on the other 21
  voc_start_scores_tl1/scores/<inv>.parquet           scores of all other inventories (kNN on all 22)

EmbeddingPredictor turns these into features: the kNN score and how sure the
seeds are of it, the similarity of a scan to its neighbours, and the scores of
linear probes on the embeddings for start, end and non-document scans
(EmbeddingProbe, stored next to the boundary model). Scans without text have
no embedding; their features are missing (the boundary model fills in the
training mean). Inventories without these files get all-missing features.

The kNN was fitted on the validated inventories, so their own scores must be
out-of-fold (the oof files); likewise the probe scores of the validated
inventories come from probes fitted without them (attach_oof_probes).
"""

import functools
import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .inventory import InventoryData
from .predictors import Predictor

logger = logging.getLogger(__name__)

EMBEDDINGS_DIR = os.environ.get("SEGMENTATION_EMBEDDINGS_DIR", "/Volumes/storage/embeddings-marijn/voc_start_scoring/output")
EMBEDDING_SUBDIR = os.path.join("embeddings", "voc_paragraph_marginalia_tl1")
SCORES_SUBDIR = "voc_start_scores_tl1"
PROBE_PATH = os.environ.get("SEGMENTATION_EMBEDDING_PROBE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "embedding_probe.npz"))
SEEDS = ("p_start_seed42", "p_start_seed43", "p_start_seed44")
PROBE_TARGETS = ("start", "end", "nondoc")
MAX_GAP = 5  # neighbours further away than this many scans do not count
WINDOW = 3  # text scans averaged before and after a possible start


def load_knn_scores(inventory_number: str, base: str = EMBEDDINGS_DIR) -> pd.DataFrame | None:
    """The kNN scores by filename: out-of-fold for the validated inventories."""
    for sub in ("oof", "scores"):
        path = os.path.join(base, SCORES_SUBDIR, sub, f"{inventory_number}.parquet")
        if os.path.exists(path):
            s = pd.read_parquet(path, columns=["scan_id", "p_start", *SEEDS])
            return s.set_index(s["scan_id"].str.split("/", n=1).str[1])
    return None


def load_embeddings(inv: InventoryData, base: str = EMBEDDINGS_DIR) -> np.ndarray | None:
    """Unit-length embedding per scan (n × 768), NaN rows for scans without text."""
    path = os.path.join(base, EMBEDDING_SUBDIR, f"{inv.inventory_number}.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    e = z["text_embeddings"].astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True).clip(1e-6)
    row = {s.split("/", 1)[1]: i for i, s in enumerate(z["scan_ids"])}
    pos = inv.scans["filename"].map(row).to_numpy(dtype=float)
    out = np.full((inv.n, e.shape[1]), np.nan, dtype=np.float32)
    has = ~np.isnan(pos)
    out[has] = e[pos[has].astype(int)]
    return out


def knn_features(inv: InventoryData, scores: pd.DataFrame | None) -> pd.DataFrame:
    fn = inv.scans["filename"].reset_index(drop=True)
    if scores is None:
        p = pd.Series(np.nan, index=fn.index)
        seeds = pd.DataFrame(np.nan, index=fn.index, columns=list(SEEDS))
    else:
        p = fn.map(scores["p_start"])
        seeds = pd.DataFrame({c: fn.map(scores[c]) for c in SEEDS})
    pc = p.clip(0.01, 0.99)
    known = p.notna()
    return pd.DataFrame(
        {
            "knn_p": p,
            "knn_logit": np.log(pc / (1 - pc)),
            "knn_zero": (p == 0).astype(float).where(known),
            "knn_all_seeds": (seeds > 0.5).all(axis=1).astype(float).where(known),
            "knn_seed_sd": seeds.std(axis=1),
            # how much higher than the best other score within two scans
            "knn_peak": p - pd.concat([p.shift(k) for k in (-2, -1, 1, 2)], axis=1).max(axis=1),
            "knn_missing": (~known).astype(float),
            "_knn_seeds_start": (seeds > 0.5).sum(axis=1).where(known),  # for display only
        }
    )


def similarity_features(E: np.ndarray | None, n: int) -> pd.DataFrame:
    """Cosine similarity to the nearest text scan before and after (within
    MAX_GAP scans), and between the mean of WINDOW text scans before scan i
    and the mean of scan i and the next text scans."""
    out = {k: np.full(n, np.nan) for k in ("emb_prev_cos", "emb_next_cos", "emb_across")}
    if E is None:
        return pd.DataFrame(out)
    text = np.flatnonzero(~np.isnan(E[:, 0]))
    for k, i in enumerate(text):
        if k > 0 and i - text[k - 1] <= MAX_GAP:
            out["emb_prev_cos"][i] = E[i] @ E[text[k - 1]]
        if k + 1 < len(text) and text[k + 1] - i <= MAX_GAP:
            out["emb_next_cos"][i] = E[i] @ E[text[k + 1]]
        before = text[max(0, k - WINDOW) : k]
        if len(before) and before[-1] >= i - MAX_GAP:
            a, b = E[before].mean(axis=0), E[text[k : k + WINDOW]].mean(axis=0)
            out["emb_across"][i] = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    return pd.DataFrame(out)


@dataclass
class EmbeddingProbe:
    """Logistic regressions on the leading principal components of the
    embeddings, one per target (start, end, nondoc); scores are logits."""

    mean: np.ndarray
    components: np.ndarray  # n_components × 768
    coef: np.ndarray  # targets × n_components
    intercept: np.ndarray  # targets

    def scores(self, E: np.ndarray | None, n: int) -> pd.DataFrame:
        out = np.full((n, len(PROBE_TARGETS)), np.nan)
        if E is not None:
            has = ~np.isnan(E[:, 0])
            if has.any():
                out[has] = ((E[has] - self.mean) @ self.components.T) @ self.coef.T + self.intercept
        return pd.DataFrame(out, columns=[f"probe_{t}" for t in PROBE_TARGETS])

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray, n_components: int = 256, C: float = 10.0) -> "EmbeddingProbe":
        """X: embeddings of text scans; Y: 0/1 targets, one column per PROBE_TARGETS.
        n_components and C chosen by cross-validation on the validated inventories:
        more components stop helping after 256; C=10 beats C=3 on starts in most
        inventories, at some cost in start recall on the (much longer) General
        Missives."""
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression

        pca = PCA(n_components, random_state=0).fit(X)
        Z = pca.transform(X)
        lrs = [LogisticRegression(C=C, max_iter=2000).fit(Z, Y[:, j]) for j in range(Y.shape[1])]
        return cls(
            pca.mean_.astype(np.float32),
            pca.components_.astype(np.float32),
            np.vstack([lr.coef_[0] for lr in lrs]),
            np.array([lr.intercept_[0] for lr in lrs]),
        )

    def save(self, path: str = PROBE_PATH):
        np.savez(path, mean=self.mean, components=self.components, coef=self.coef, intercept=self.intercept)

    @classmethod
    def load(cls, path: str = PROBE_PATH) -> "EmbeddingProbe | None":
        if not os.path.exists(path):
            return None
        z = np.load(path)
        return cls(z["mean"], z["components"], z["coef"], z["intercept"])


@functools.cache
def default_probe() -> EmbeddingProbe | None:
    probe = EmbeddingProbe.load()
    if probe is None:
        logger.warning("No embedding probe at %s (segmentation fit writes it): probe features are missing", PROBE_PATH)
    return probe


@functools.cache
def _check_dir(base: str):
    if not os.path.isdir(base):
        logger.warning("Embeddings folder %s not found (SEGMENTATION_EMBEDDINGS_DIR): embedding features are missing", base)


class EmbeddingPredictor(Predictor):
    """kNN start scores, neighbour similarities and probe scores (see module docstring)."""

    name = "embeddings"

    def __init__(self, base: str = EMBEDDINGS_DIR, probe: EmbeddingProbe | None = None):
        _check_dir(base)
        self.base = base
        self.probe = probe or default_probe()

    def compute(self, inv):
        E = load_embeddings(inv, self.base)
        if self.probe:
            probe = self.probe.scores(E, inv.n)
        else:
            probe = pd.DataFrame(np.nan, index=range(inv.n), columns=[f"probe_{t}" for t in PROBE_TARGETS])
        return pd.concat([knn_features(inv, load_knn_scores(inv.inventory_number, self.base)), similarity_features(E, inv.n), probe], axis=1)
