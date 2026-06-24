"""Refresh web_app/_runtime.json so the Flask app picks up the latest hierarchy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import DEFAULT_OUT_ROOT


def _cut_entry(out_root: Path, cut_name: str) -> dict:
    winner_path = out_root / "cuts" / cut_name / "ensemble" / "winner.json"
    if not winner_path.exists():
        raise SystemExit(f"Missing winner.json for cut '{cut_name}': {winner_path}")
    winner_payload = json.loads(winner_path.read_text(encoding="utf-8"))
    winner = winner_payload["decision"]
    members = [
        {"backbone": m["backbone"], "fold": m["fold"], "ckpt": str(Path(m["ckpt"]).resolve())}
        for m in winner["members"]
    ]
    meta_path = winner_payload.get("meta_path")
    return {
        "kind": winner["chosen"],
        "members": members,
        "cv_dir": str((out_root / "cuts" / cut_name / "cv").resolve()),
        "ensemble_dir": str((out_root / "cuts" / cut_name / "ensemble").resolve()),
        "meta_path": str(Path(meta_path).resolve()) if meta_path else None,
    }


def _pick_combiner(out_root: Path) -> dict | None:
    """Among the trained combiner feature sets, pick the one with the highest OOF
    macro-F1 whose model file exists. 'all' usually wins; that is fine — the web
    will load whatever cuts it needs."""
    best = None
    for fs in ("all", "topology3"):
        bp = out_root / "combiner" / fs / "best.json"
        if not bp.exists():
            continue
        cb = json.loads(bp.read_text(encoding="utf-8"))
        mp = cb.get("model_path")
        if not mp or not Path(mp).exists():
            continue
        oof = float(cb.get("oof_macro_f1", float("-inf")))
        if best is None or oof > best[0]:
            best = (oof, cb)
    return best[1] if best else None


def main():
    p = argparse.ArgumentParser(description="Sync web_app runtime config to latest hierarchy.")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--web-app", type=Path, default=Path("web_app"))
    args = p.parse_args()

    best_topo_path = args.out_root / "hierarchy" / "best_topology.json"
    if not best_topo_path.exists():
        raise SystemExit(f"No best_topology.json at {best_topo_path}; run search_hierarchy first.")
    payload = json.loads(best_topo_path.read_text(encoding="utf-8"))

    # The web needs: the topology's cuts (routing fallback + Grad-CAM) PLUS the
    # chosen combiner's cuts (which may be all 10). Load the union.
    cuts_to_load: set[str] = set(payload["cuts_used"])
    combiner = _pick_combiner(args.out_root)
    if combiner is not None:
        cuts_to_load |= set(combiner.get("cuts_used", []))

    cuts_payload = {cn: _cut_entry(args.out_root, cn) for cn in sorted(cuts_to_load)}

    runtime = {
        "topology": payload,
        "cuts": cuts_payload,
        "out_root": str(args.out_root.resolve()),
    }

    if combiner is not None:
        cb_cuts = list(combiner.get("cuts_used", []))
        runtime["combiner"] = {
            "featureset": combiner.get("featureset"),
            "best_model": combiner.get("best_model"),
            "model_path": str(Path(combiner["model_path"]).resolve()),
            "cuts": cb_cuts,  # feature column order
        }
        print(f"   combiner: {combiner.get('featureset')} / {combiner.get('best_model')} "
              f"({len(cb_cuts)} cuts, oof={combiner.get('oof_macro_f1'):.4f})")
    else:
        print("   combiner: none found (web will use hierarchy hard routing)")

    runtime_path = args.web_app / "_runtime.json"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(json.dumps(runtime, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {runtime_path}")
    print(f"   topology: {payload['name']} {payload['description']}")
    print(f"   cuts loaded: {sorted(cuts_payload.keys())}")


if __name__ == "__main__":
    main()
