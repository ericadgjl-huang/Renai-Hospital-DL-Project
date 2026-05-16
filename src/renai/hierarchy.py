"""Exhaustive hierarchy search over the 5 binary-tree topologies on 4 stages.

A topology is a binary tree whose leaves are stages 1..4 in increasing order.
The Catalan number for 4 leaves is 5, so there are exactly 5 ordered topologies:

    T1: ((1,2),(3,4))             cuts: 12_vs_34, 1_vs_2,  3_vs_4
    T2: (1,(2,(3,4)))             cuts: 1_vs_234, 2_vs_34, 3_vs_4
    T3: (1,((2,3),4))             cuts: 1_vs_234, 23_vs_4, 2_vs_3
    T4: ((1,(2,3)),4)             cuts: 123_vs_4, 1_vs_23, 2_vs_3
    T5: (((1,2),3),4)             cuts: 123_vs_4, 12_vs_3, 1_vs_2

For each topology we route a sample through the binary classifiers and produce
a 4-class prediction.  The topology that maximizes test 4-class macro-F1 wins.

This module does *not* train models — it consumes the already-trained per-cut
ensemble winners (single/voting/stacking) and just runs inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import Cut, make_4class_eval_loader, make_outer_split
from .eval import (
    binary_metrics,
    dump_json,
    save_classification_report,
    save_confusion_matrix,
)
from .models import create_model
from .seed import SEED, set_seed


# ----- Topology definitions -------------------------------------------------

@dataclass(frozen=True)
class Topology:
    name: str
    description: str
    # Each routing rule: ("cut_name", left_subset, right_subset)
    # left_subset/right_subset use 1-indexed stage numbers.
    # The left branch corresponds to the cut's class-0 prediction.
    rules: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]


TOPOLOGIES: dict[str, Topology] = {
    "T1": Topology(
        "T1", "((1,2),(3,4))",
        rules=(
            ("12_vs_34", (1, 2), (3, 4)),
            ("1_vs_2",   (1,),   (2,)),
            ("3_vs_4",   (3,),   (4,)),
        ),
    ),
    "T2": Topology(
        "T2", "(1,(2,(3,4)))",
        rules=(
            ("1_vs_234", (1,),   (2, 3, 4)),
            ("2_vs_34",  (2,),   (3, 4)),
            ("3_vs_4",   (3,),   (4,)),
        ),
    ),
    "T3": Topology(
        "T3", "(1,((2,3),4))",
        rules=(
            ("1_vs_234", (1,),   (2, 3, 4)),
            ("23_vs_4",  (2, 3), (4,)),
            ("2_vs_3",   (2,),   (3,)),
        ),
    ),
    "T4": Topology(
        "T4", "((1,(2,3)),4)",
        rules=(
            ("123_vs_4", (1, 2, 3), (4,)),
            ("1_vs_23",  (1,),      (2, 3)),
            ("2_vs_3",   (2,),      (3,)),
        ),
    ),
    "T5": Topology(
        "T5", "(((1,2),3),4)",
        rules=(
            ("123_vs_4", (1, 2, 3), (4,)),
            ("12_vs_3",  (1, 2),    (3,)),
            ("1_vs_2",   (1,),      (2,)),
        ),
    ),
}


# ----- Per-cut binary predictor (post-ensemble) -----------------------------

class CutPredictor:
    """Loads whatever winner the cut chose (single / voting / stacking) and
    exposes a uniform `prob_class1(x)` interface."""

    def __init__(self, cut_dir: Path, device: str):
        self.cut_dir = cut_dir
        self.device = device

        winner_path = cut_dir / "ensemble" / "winner.json"
        if not winner_path.exists():
            raise FileNotFoundError(f"Missing winner.json at {winner_path}")
        info = json.loads(winner_path.read_text(encoding="utf-8"))
        self.kind: str = info["decision"]["chosen"]
        self.members: list[str] = list(info["decision"]["chosen_members"])
        self.backbone: str | None = info["decision"]["chosen_backbone"]

        self._models = []
        for bb in (self.members if self.kind != "single" else [self.backbone]):
            ckpt = cut_dir / "final" / bb / f"best_{bb}.pth"
            m = create_model(bb, num_classes=2).to(device)
            m.load_state_dict(torch.load(ckpt, map_location=device))
            m.eval()
            self._models.append((bb, m))

        self._meta = None
        if self.kind == "stacking":
            self._meta = joblib.load(cut_dir / "ensemble" / "meta_logreg.pkl")

    @torch.no_grad()
    def prob_class1(self, x: torch.Tensor) -> np.ndarray:
        """Return P(class=1) per sample. Class 1 == cut.positives_one."""
        if self.kind == "single":
            _, m = self._models[0]
            p = F.softmax(m(x.to(self.device)), dim=1).cpu().numpy()
            return p[:, 1]

        # Stacking / voting both need full softmax from each member.
        stacks = []
        for _, m in self._models:
            p = F.softmax(m(x.to(self.device)), dim=1).cpu().numpy()
            stacks.append(p)
        if self.kind == "voting":
            avg = np.mean(stacks, axis=0)
            return avg[:, 1]
        if self.kind == "stacking" and self._meta is not None:
            X = np.concatenate(stacks, axis=1)
            return self._meta.predict_proba(X)[:, 1]
        raise RuntimeError(f"Unknown predictor kind {self.kind}")


# ----- Hierarchy inference --------------------------------------------------

def topology_required_cuts(topo: Topology) -> set[str]:
    return {r[0] for r in topo.rules}


def predict_topology(
    topo: Topology,
    cut_predictors: dict[str, CutPredictor],
    test_loader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true_1based, y_pred_1based) where labels are 1..4."""
    # Pre-compute per-cut probabilities for every test sample, in order.
    per_cut: dict[str, np.ndarray] = {}
    y_true: list[int] = []
    needed = topology_required_cuts(topo)

    # Run each model once over the full loader to collect probabilities.
    # We can iterate the loader multiple times (it is shuffle=False).
    first_pass = True
    for cut_name in needed:
        pred = cut_predictors[cut_name]
        probs = []
        for imgs, labels in test_loader:
            probs.append(pred.prob_class1(imgs))
            if first_pass:
                y_true.extend(int(l) + 1 for l in labels.numpy().tolist())
        per_cut[cut_name] = np.concatenate(probs)
        first_pass = False

    y_true_arr = np.asarray(y_true, dtype=np.int64)

    # Apply routing rules to compute the final class.
    n = len(y_true_arr)
    pred_class = np.zeros(n, dtype=np.int64)

    rule_a = topo.rules[0]   # top split
    cutA, leftA, rightA = rule_a
    pA1 = per_cut[cutA]      # P(class=1) -> right subset

    # Topologies are all of "left vs right" where left or right is a singleton
    # and the other is decomposed by a second cut.  We handle the general case
    # by recursing through the rule list.

    def resolve(subset: tuple[int, ...], i: int, rules: list) -> int:
        if len(subset) == 1:
            return subset[0]
        # Find a rule whose union equals `subset`.
        for r in rules:
            cn, L, R = r
            if tuple(sorted(L + R)) == tuple(sorted(subset)):
                p_right = per_cut[cn][i]
                go_right = p_right >= 0.5
                next_subset = R if go_right else L
                next_rules = [rr for rr in rules if rr is not r]
                return resolve(next_subset, i, next_rules)
        # Should not happen if topology is well-formed.
        return subset[0]

    for i in range(n):
        go_right = pA1[i] >= 0.5
        subset = rightA if go_right else leftA
        remaining_rules = [r for r in topo.rules[1:]]
        pred_class[i] = resolve(subset, i, remaining_rules)

    return y_true_arr, pred_class


def search_best_topology(
    cuts: dict[str, Cut],
    data_root: Path,
    out_root: Path,
    splits_dir: Path,
    device: str = "cuda",
    batch_size: int = 16,
) -> dict:
    set_seed(SEED)
    outer = make_outer_split(data_root, splits_dir / "outer_split.json")
    test_loader = make_4class_eval_loader(data_root, outer["test_idx"], batch_size=batch_size)

    # Build all needed cut predictors lazily — only those referenced by some
    # topology AND whose ensemble step has already been run.
    predictors: dict[str, CutPredictor] = {}
    available_cuts: set[str] = set()
    for cn in {r[0] for t in TOPOLOGIES.values() for r in t.rules}:
        cut_dir = out_root / "cuts" / cn
        if (cut_dir / "ensemble" / "winner.json").exists():
            predictors[cn] = CutPredictor(cut_dir, device)
            available_cuts.add(cn)

    rows = []
    hier_dir = out_root / "hierarchy"
    hier_dir.mkdir(parents=True, exist_ok=True)

    for topo in TOPOLOGIES.values():
        needed = topology_required_cuts(topo)
        if not needed.issubset(available_cuts):
            missing = sorted(needed - available_cuts)
            print(f"  topology {topo.name} skipped (missing cuts: {missing})", flush=True)
            rows.append({
                "topology": topo.name, "description": topo.description,
                "status": "skipped", "missing_cuts": ",".join(missing),
                "macro_f1": float("nan"), "accuracy": float("nan"),
            })
            continue

        y_true, y_pred = predict_topology(topo, predictors, test_loader, device)
        m = binary_metrics(y_true, y_pred)  # repurposed; multiclass works for accuracy/macro_f1
        # For multiclass we need a proper macro_f1 calc:
        from sklearn.metrics import f1_score as _f1
        m["macro_f1"] = float(_f1(y_true, y_pred, average="macro", zero_division=0))
        m["weighted_f1"] = float(_f1(y_true, y_pred, average="weighted", zero_division=0))

        topo_dir = hier_dir / topo.name
        topo_dir.mkdir(parents=True, exist_ok=True)
        save_confusion_matrix(
            y_true - 1, y_pred - 1,
            ["stage_1", "stage_2", "stage_3", "stage_4"],
            topo_dir / "confusion_matrix_test.png",
            title=f"{topo.name} {topo.description}",
        )
        save_classification_report(
            y_true - 1, y_pred - 1,
            ["stage_1", "stage_2", "stage_3", "stage_4"],
            topo_dir / "classification_report_test.txt",
        )
        dump_json({"topology": topo.name, **m}, topo_dir / "metrics.json")

        rows.append({
            "topology": topo.name, "description": topo.description,
            "status": "ok", "missing_cuts": "",
            **{k: m[k] for k in ("accuracy", "macro_f1", "weighted_f1")},
        })

    df = pd.DataFrame(rows)
    df_ok = df[df["status"] == "ok"].dropna(subset=["macro_f1"])
    best_row = df_ok.sort_values("macro_f1", ascending=False).iloc[0] if len(df_ok) else None

    df.to_csv(hier_dir / "search_results.csv", index=False, encoding="utf-8-sig")

    if best_row is not None:
        topo = TOPOLOGIES[str(best_row["topology"])]
        winner_payload = {
            "name": topo.name,
            "description": topo.description,
            "macro_f1": float(best_row["macro_f1"]),
            "accuracy": float(best_row["accuracy"]),
            "rules": [
                {"cut": r[0], "left": list(r[1]), "right": list(r[2])}
                for r in topo.rules
            ],
            "cuts_used": sorted(topology_required_cuts(topo)),
        }
        dump_json(winner_payload, hier_dir / "best_topology.json")
        # Also a YAML form for the web app to consume.
        try:
            import yaml  # type: ignore
            (hier_dir / "best_topology.yaml").write_text(
                yaml.safe_dump(winner_payload, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        except Exception:
            pass
        return winner_payload

    return {"name": None, "description": "no topology evaluated"}
