"""Exhaustive hierarchy search with OOF-only selection (v2.1).

For 4 ordered stages there are C(3) = 5 binary-tree topologies; each routes
samples through 3 binary cuts.  Cuts are pre-computed by `ensemble.py` and
saved as:

    outputs/cuts/<cut>/oof/all_train_val.npz   per-backbone OOF probs + labels
    outputs/cuts/<cut>/oof/oof_p_class1.npy    P(class=1) per train_val sample
                                               under the cut's chosen strategy
                                               (already unbiased)
    outputs/cuts/<cut>/ensemble/winner.json
    outputs/cuts/<cut>/ensemble/meta_logreg.pkl
    outputs/cuts/<cut>/final/<bb>/best_<bb>.pth   final retrained ckpts (used
                                                  ONLY for the test set pass)

Topology search procedure
-------------------------
1. For each topology, route every train_val sample through the tree using
   the pre-computed OOF P(class=1) from each required cut.  Compute the
   resulting 4-class OOF macro-F1.
2. Pick the winning topology by OOF macro-F1.  **No test set is touched.**
3. ONLY for the winner: run the full inference path on the 20% outer test
   set using the final retrained ckpts + (for stacking) the deployment meta.
   Save the resulting 4-class CM + classification report + metrics — that is
   the unbiased final number to report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from .data import (
    Cut,
    filter_indices_for_cut,
    get_4class_labels,
    make_4class_eval_loader,
    make_outer_split,
)
from .eval import (
    dump_json,
    save_classification_report,
    save_confusion_matrix,
)
from .models import create_model
from .seed import SEED, set_seed


# ---------------------------------------------------------------------------
# Topology definitions (mirrored in configs/topologies.yaml)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Topology:
    name: str
    description: str
    rules: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]


TOPOLOGIES: dict[str, Topology] = {
    "T1": Topology("T1", "((1,2),(3,4))", rules=(
        ("12_vs_34", (1, 2),    (3, 4)),
        ("1_vs_2",   (1,),      (2,)),
        ("3_vs_4",   (3,),      (4,)),
    )),
    "T2": Topology("T2", "(1,(2,(3,4)))", rules=(
        ("1_vs_234", (1,),      (2, 3, 4)),
        ("2_vs_34",  (2,),      (3, 4)),
        ("3_vs_4",   (3,),      (4,)),
    )),
    "T3": Topology("T3", "(1,((2,3),4))", rules=(
        ("1_vs_234", (1,),      (2, 3, 4)),
        ("23_vs_4",  (2, 3),    (4,)),
        ("2_vs_3",   (2,),      (3,)),
    )),
    "T4": Topology("T4", "((1,(2,3)),4)", rules=(
        ("123_vs_4", (1, 2, 3), (4,)),
        ("1_vs_23",  (1,),      (2, 3)),
        ("2_vs_3",   (2,),      (3,)),
    )),
    "T5": Topology("T5", "(((1,2),3),4)", rules=(
        ("123_vs_4", (1, 2, 3), (4,)),
        ("12_vs_3",  (1, 2),    (3,)),
        ("1_vs_2",   (1,),      (2,)),
    )),
}


def topology_required_cuts(topo: Topology) -> set[str]:
    return {r[0] for r in topo.rules}


# ---------------------------------------------------------------------------
# Tree routing
# ---------------------------------------------------------------------------

def _route_tree(
    topo: Topology,
    p_class1_per_cut: dict[str, np.ndarray],   # cut_name -> (N,) array of P(class=1)
    n_samples: int,
) -> np.ndarray:
    """Route each of n_samples through the topology tree using per-cut
    P(class=1).  Returns 1..4 predicted class per sample."""
    rules = list(topo.rules)

    def resolve_for_one(i: int, subset: tuple[int, ...], remaining: list) -> int:
        if len(subset) == 1:
            return subset[0]
        for r in remaining:
            cn, L, R = r
            if tuple(sorted(L + R)) == tuple(sorted(subset)):
                p_right = float(p_class1_per_cut[cn][i])
                next_subset = R if p_right >= 0.5 else L
                rest = [rr for rr in remaining if rr is not r]
                return resolve_for_one(i, next_subset, rest)
        return subset[0]  # malformed topology safety net

    pred = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        pred[i] = resolve_for_one(i, (1, 2, 3, 4), list(rules))
    return pred


# ---------------------------------------------------------------------------
# OOF topology evaluation (selection)
# ---------------------------------------------------------------------------

def _load_oof_p_class1(out_root: Path, cut_name: str) -> np.ndarray:
    p = out_root / "cuts" / cut_name / "oof" / "oof_p_class1.npy"
    if not p.exists():
        raise FileNotFoundError(
            f"missing {p} — run scripts/05_build_ensemble.py first"
        )
    return np.load(p)


def _load_train_val_labels(out_root: Path, any_cut_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load y_4class (0..3) and train_val_idx from any cut's OOF npz (they all
    share the same outer split + sample ordering)."""
    npz = np.load(out_root / "cuts" / any_cut_name / "oof" / "all_train_val.npz")
    return npz["y_4class"], npz["all_train_val_idx"]


# ---------------------------------------------------------------------------
# Test inference (only for winning topology)
# ---------------------------------------------------------------------------

class TestCutPredictor:
    """Loads winner.json and serves P(class=1) for arbitrary test images.

    Uses the FINAL retrained checkpoints (trained on full 80%) for base
    predictions; for stacking, also loads the meta classifier trained on the
    full OOF matrix."""

    def __init__(self, cut_dir: Path, device: str):
        self.cut_dir = cut_dir
        self.device = device
        info = json.loads((cut_dir / "ensemble" / "winner.json").read_text(encoding="utf-8"))
        decision = info["decision"]
        self.kind: str = decision["chosen"]
        self.members: list[str] = list(decision["chosen_members"])
        self.backbone: str | None = decision["chosen_backbone"]

        self._models: list[tuple[str, torch.nn.Module]] = []
        for bb in (self.members if self.kind != "single" else [self.backbone]):
            ckpt = cut_dir / "final" / bb / f"best_{bb}.pth"
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"missing final ckpt {ckpt} — ensure scripts/04 ran fully"
                )
            m = create_model(bb, num_classes=2).to(device)
            m.load_state_dict(torch.load(ckpt, map_location=device))
            m.eval()
            self._models.append((bb, m))

        self._meta = (
            joblib.load(cut_dir / "ensemble" / "meta_logreg.pkl")
            if self.kind == "stacking" else None
        )

    @torch.no_grad()
    def prob_class1_batch(self, x: torch.Tensor) -> np.ndarray:
        x = x.to(self.device)
        stacks = []
        for _, m in self._models:
            stacks.append(F.softmax(m(x), dim=1).cpu().numpy())
        if self.kind == "single":
            return stacks[0][:, 1]
        if self.kind == "voting":
            return np.mean(stacks, axis=0)[:, 1]
        if self.kind == "stacking" and self._meta is not None:
            X = np.concatenate(stacks, axis=1)
            return self._meta.predict_proba(X)[:, 1]
        raise RuntimeError(f"unknown predictor kind {self.kind}")


def _compute_test_p_class1(
    cut_predictors: dict[str, TestCutPredictor],
    needed_cuts: set[str],
    test_loader,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Run each cut predictor over the test loader once; return per-cut
    P(class=1) arrays plus the corresponding 1..4 ground-truth labels."""
    y_true: list[int] = []
    p_by_cut: dict[str, list[np.ndarray]] = {cn: [] for cn in needed_cuts}
    for imgs, labels in test_loader:
        # Collect labels EVERY batch (previous version had a `first` flag that
        # only kept batch-0 labels, but predictions accumulated across all
        # batches — predictions and labels desynced after batch 0).
        y_true.extend(int(l) + 1 for l in labels.numpy().tolist())
        for cn in needed_cuts:
            p_by_cut[cn].append(cut_predictors[cn].prob_class1_batch(imgs))
    return (
        {cn: np.concatenate(p_by_cut[cn]) for cn in needed_cuts},
        np.asarray(y_true, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_best_topology(
    cuts: dict[str, Cut],
    data_root: Path,
    out_root: Path,
    splits_dir: Path,
    device: str = "cuda",
    batch_size: int = 16,
) -> dict:
    set_seed(SEED)
    hier_dir = out_root / "hierarchy"
    hier_dir.mkdir(parents=True, exist_ok=True)

    needed_cuts = {r[0] for t in TOPOLOGIES.values() for r in t.rules}

    # ---- Phase 1: evaluate every topology on OOF (no test set) ----
    available_cuts: set[str] = set()
    oof_p_per_cut: dict[str, np.ndarray] = {}
    for cn in needed_cuts:
        try:
            oof_p_per_cut[cn] = _load_oof_p_class1(out_root, cn)
            available_cuts.add(cn)
        except FileNotFoundError as e:
            print(f"  [hierarchy] {e}", flush=True)

    if not available_cuts:
        raise RuntimeError(
            "no cut OOF files found; run scripts/05_build_ensemble.py first"
        )

    # All cut OOF arrays must align with the same train_val ordering — pick
    # one to read y_4class, then sanity-check the others have matching length.
    sample_cut = next(iter(available_cuts))
    y_4class, train_val_idx = _load_train_val_labels(out_root, sample_cut)
    n_train_val = len(y_4class)
    for cn in available_cuts:
        if len(oof_p_per_cut[cn]) != n_train_val:
            raise RuntimeError(
                f"cut {cn} OOF length {len(oof_p_per_cut[cn])} ≠ {n_train_val}"
            )

    y_true_train_val = y_4class + 1  # 1..4

    rows = []
    for topo in TOPOLOGIES.values():
        needed = topology_required_cuts(topo)
        if not needed.issubset(available_cuts):
            missing = sorted(needed - available_cuts)
            print(f"  topology {topo.name} skipped (missing cuts: {missing})", flush=True)
            rows.append({
                "topology": topo.name, "description": topo.description,
                "status": "skipped", "missing_cuts": ",".join(missing),
                "oof_macro_f1": float("nan"), "oof_accuracy": float("nan"),
            })
            continue

        pred_train_val = _route_tree(topo, oof_p_per_cut, n_train_val)
        oof_macro_f1 = float(f1_score(y_true_train_val, pred_train_val, average="macro", zero_division=0))
        oof_weighted_f1 = float(f1_score(y_true_train_val, pred_train_val, average="weighted", zero_division=0))
        oof_acc = float(accuracy_score(y_true_train_val, pred_train_val))

        topo_dir = hier_dir / topo.name
        topo_dir.mkdir(parents=True, exist_ok=True)
        save_confusion_matrix(
            y_true_train_val - 1, pred_train_val - 1,
            ["stage_1", "stage_2", "stage_3", "stage_4"],
            topo_dir / "confusion_matrix_oof.png",
            title=f"{topo.name} {topo.description} (OOF)",
        )
        save_classification_report(
            y_true_train_val - 1, pred_train_val - 1,
            ["stage_1", "stage_2", "stage_3", "stage_4"],
            topo_dir / "classification_report_oof.txt",
        )
        dump_json({
            "topology": topo.name,
            "description": topo.description,
            "oof_macro_f1": oof_macro_f1,
            "oof_weighted_f1": oof_weighted_f1,
            "oof_accuracy": oof_acc,
        }, topo_dir / "metrics_oof.json")

        rows.append({
            "topology": topo.name, "description": topo.description,
            "status": "ok", "missing_cuts": "",
            "oof_macro_f1": oof_macro_f1,
            "oof_weighted_f1": oof_weighted_f1,
            "oof_accuracy": oof_acc,
        })

    import pandas as pd  # local import keeps module fast to import
    df = pd.DataFrame(rows)
    df_ok = df[df["status"] == "ok"].dropna(subset=["oof_macro_f1"])
    if not len(df_ok):
        df.to_csv(hier_dir / "search_results.csv", index=False, encoding="utf-8-sig")
        return {"name": None, "description": "no topology evaluated"}

    best_row = df_ok.sort_values("oof_macro_f1", ascending=False).iloc[0]
    best_topo_name = str(best_row["topology"])
    best_topo = TOPOLOGIES[best_topo_name]
    print(
        f"\n  >> OOF winner: {best_topo_name} {best_topo.description} "
        f"oof_macro_f1={best_row['oof_macro_f1']:.4f}",
        flush=True,
    )

    # ---- Phase 2: ONE test-set pass for the winner ----
    outer = make_outer_split(data_root, splits_dir / "outer_split.json")
    test_loader = make_4class_eval_loader(data_root, outer["test_idx"], batch_size=batch_size)

    winner_needed = topology_required_cuts(best_topo)
    test_predictors = {
        cn: TestCutPredictor(out_root / "cuts" / cn, device) for cn in winner_needed
    }
    test_p_per_cut, y_true_test = _compute_test_p_class1(test_predictors, winner_needed, test_loader)
    pred_test = _route_tree(best_topo, test_p_per_cut, len(y_true_test))

    test_macro_f1 = float(f1_score(y_true_test, pred_test, average="macro", zero_division=0))
    test_weighted_f1 = float(f1_score(y_true_test, pred_test, average="weighted", zero_division=0))
    test_acc = float(accuracy_score(y_true_test, pred_test))

    winner_dir = hier_dir / best_topo_name
    save_confusion_matrix(
        y_true_test - 1, pred_test - 1,
        ["stage_1", "stage_2", "stage_3", "stage_4"],
        winner_dir / "confusion_matrix_test.png",
        title=f"{best_topo_name} {best_topo.description} (TEST — single pass)",
    )
    save_classification_report(
        y_true_test - 1, pred_test - 1,
        ["stage_1", "stage_2", "stage_3", "stage_4"],
        winner_dir / "classification_report_test.txt",
    )

    final_payload = {
        "name": best_topo_name,
        "description": best_topo.description,
        "selection_metric": "4-class OOF macro-F1 (no test set used for selection)",
        "oof_macro_f1": float(best_row["oof_macro_f1"]),
        "oof_accuracy": float(best_row["oof_accuracy"]),
        "test_macro_f1": test_macro_f1,
        "test_weighted_f1": test_weighted_f1,
        "test_accuracy": test_acc,
        "rules": [
            {"cut": r[0], "left": list(r[1]), "right": list(r[2])}
            for r in best_topo.rules
        ],
        "cuts_used": sorted(winner_needed),
    }
    dump_json(final_payload, hier_dir / "best_topology.json")
    dump_json(final_payload, winner_dir / "metrics_test.json")

    try:
        import yaml  # type: ignore
        (hier_dir / "best_topology.yaml").write_text(
            yaml.safe_dump(final_payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except Exception:
        pass

    df.to_csv(hier_dir / "search_results.csv", index=False, encoding="utf-8-sig")
    return final_payload
