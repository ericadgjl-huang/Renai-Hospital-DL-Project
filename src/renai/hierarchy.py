"""Exhaustive hierarchy search over the 5 binary-tree topologies on 4 stages.

A topology is a binary tree whose leaves are stages 1..4 in increasing order.
The Catalan number for 4 leaves is 5, so there are exactly 5 ordered topologies:

    T1: ((1,2),(3,4))             cuts: 12_vs_34, 1_vs_2,  3_vs_4
    T2: (1,(2,(3,4)))             cuts: 1_vs_234, 2_vs_34, 3_vs_4
    T3: (1,((2,3),4))             cuts: 1_vs_234, 23_vs_4, 2_vs_3
    T4: ((1,(2,3)),4)             cuts: 123_vs_4, 1_vs_23, 2_vs_3
    T5: (((1,2),3),4)             cuts: 123_vs_4, 12_vs_3, 1_vs_2

For each topology we route a sample through the per-cut binary predictors and
produce a 4-class prediction.  The topology that maximizes test 4-class
macro-F1 wins.

This module does *not* train models — every cut's binary predictor is the
soft-vote of its 25 per-fold checkpoints (see `renai.ensemble`)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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


# ----- Per-cut fold-voting predictor ----------------------------------------

class CutPredictor:
    """Soft-vote of every per-fold best ckpt that ensemble step found for
    this cut.  Members are read from outputs/cuts/<cut>/ensemble/winner.json."""

    def __init__(self, cut_dir: Path, device: str):
        self.cut_dir = cut_dir
        self.device = device

        winner_path = cut_dir / "ensemble" / "winner.json"
        if not winner_path.exists():
            raise FileNotFoundError(f"Missing winner.json at {winner_path}")
        info = json.loads(winner_path.read_text(encoding="utf-8"))
        self.members: list[dict] = list(info["decision"]["members"])

        print(f"  [load] {cut_dir.name}: loading {len(self.members)} ckpts on {device}", flush=True)
        self._models: list[tuple[str, torch.nn.Module]] = []
        for k, entry in enumerate(self.members, 1):
            bb = entry["backbone"]
            ckpt = Path(entry["ckpt"])
            m = create_model(bb, num_classes=2).to(device)
            m.load_state_dict(torch.load(ckpt, map_location=device))
            m.eval()
            self._models.append((bb, m))
            if k % 5 == 0 or k == len(self.members):
                print(f"    .. {k}/{len(self.members)} loaded", flush=True)

    @torch.no_grad()
    def prob_class1(self, x: torch.Tensor) -> np.ndarray:
        """Mean P(class=1) across members for every sample in `x`."""
        sums: np.ndarray | None = None
        for _, m in self._models:
            p = F.softmax(m(x.to(self.device)), dim=1).cpu().numpy()
            if sums is None:
                sums = np.zeros_like(p)
            sums += p
        if sums is None:
            return np.array([])
        avg = sums / float(len(self._models))
        return avg[:, 1]


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
    per_cut: dict[str, np.ndarray] = {}
    y_true: list[int] = []
    needed = topology_required_cuts(topo)

    first_pass = True
    for cut_name in sorted(needed):
        pred = cut_predictors[cut_name]
        n_models = len(pred._models)
        n_batches = len(test_loader)
        print(
            f"    [{topo.name}] cut={cut_name}: {n_models} models x {n_batches} batches",
            flush=True,
        )
        probs = []
        for bi, (imgs, labels) in enumerate(test_loader, 1):
            probs.append(pred.prob_class1(imgs))
            if first_pass:
                y_true.extend(int(l) + 1 for l in labels.numpy().tolist())
            if bi % 10 == 0 or bi == n_batches:
                print(f"      .. batch {bi}/{n_batches}", flush=True)
        per_cut[cut_name] = np.concatenate(probs)
        first_pass = False

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    n = len(y_true_arr)
    pred_class = np.zeros(n, dtype=np.int64)

    rule_a = topo.rules[0]
    _, leftA, rightA = rule_a

    def resolve(subset: tuple[int, ...], i: int, rules: list) -> int:
        if len(subset) == 1:
            return subset[0]
        for r in rules:
            cn, L, R = r
            if tuple(sorted(L + R)) == tuple(sorted(subset)):
                p_right = per_cut[cn][i]
                go_right = p_right >= 0.5
                next_subset = R if go_right else L
                next_rules = [rr for rr in rules if rr is not r]
                return resolve(next_subset, i, next_rules)
        return subset[0]

    pA1 = per_cut[rule_a[0]]
    for i in range(n):
        go_right = pA1[i] >= 0.5
        subset = rightA if go_right else leftA
        remaining_rules = list(topo.rules[1:])
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
    print(f"[hierarchy] device={device}", flush=True)
    outer = make_outer_split(data_root, splits_dir / "outer_split.json")
    test_loader = make_4class_eval_loader(data_root, outer["test_idx"], batch_size=batch_size)
    print(
        f"[hierarchy] test_loader: {len(test_loader.dataset)} samples / "
        f"{len(test_loader)} batches",
        flush=True,
    )

    all_needed = {r[0] for t in TOPOLOGIES.values() for r in t.rules}
    print(f"[hierarchy] loading predictors for up to {len(all_needed)} cuts ...", flush=True)
    predictors: dict[str, CutPredictor] = {}
    available_cuts: set[str] = set()
    for cn in sorted(all_needed):
        cut_dir = out_root / "cuts" / cn
        if (cut_dir / "ensemble" / "winner.json").exists():
            predictors[cn] = CutPredictor(cut_dir, device)
            available_cuts.add(cn)
        else:
            print(f"  [skip] {cn}: no ensemble/winner.json", flush=True)
    print(f"[hierarchy] {len(predictors)}/{len(all_needed)} cuts ready", flush=True)

    rows = []
    hier_dir = out_root / "hierarchy"
    hier_dir.mkdir(parents=True, exist_ok=True)

    for topo in TOPOLOGIES.values():
        print(f"\n[hierarchy] === topology {topo.name} {topo.description} ===", flush=True)
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
        m = binary_metrics(y_true, y_pred)
        from sklearn.metrics import f1_score as _f1
        m["macro_f1"] = float(_f1(y_true, y_pred, average="macro", zero_division=0))
        m["weighted_f1"] = float(_f1(y_true, y_pred, average="weighted", zero_division=0))
        print(
            f"  [{topo.name}] acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}",
            flush=True,
        )

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
