"""Refresh web_app/_runtime.json so the Flask app picks up the latest hierarchy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import DEFAULT_OUT_ROOT


def main():
    p = argparse.ArgumentParser(description="Sync web_app runtime config to latest hierarchy.")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--web-app", type=Path, default=Path("web_app"))
    args = p.parse_args()

    best_topo_path = args.out_root / "hierarchy" / "best_topology.json"
    if not best_topo_path.exists():
        raise SystemExit(f"No best_topology.json at {best_topo_path}; run search_hierarchy first.")
    payload = json.loads(best_topo_path.read_text(encoding="utf-8"))

    cuts_payload = {}
    for cut_name in payload["cuts_used"]:
        winner_path = args.out_root / "cuts" / cut_name / "ensemble" / "winner.json"
        if not winner_path.exists():
            raise SystemExit(f"Missing winner.json for cut '{cut_name}': {winner_path}")
        winner_payload = json.loads(winner_path.read_text(encoding="utf-8"))
        winner = winner_payload["decision"]
        members = [
            {
                "backbone": m["backbone"],
                "fold": m["fold"],
                "ckpt": str(Path(m["ckpt"]).resolve()),
            }
            for m in winner["members"]
        ]
        meta_path = winner_payload.get("meta_path")
        cuts_payload[cut_name] = {
            "kind": winner["chosen"],
            "members": members,
            "cv_dir": str((args.out_root / "cuts" / cut_name / "cv").resolve()),
            "ensemble_dir": str((args.out_root / "cuts" / cut_name / "ensemble").resolve()),
            "meta_path": str(Path(meta_path).resolve()) if meta_path else None,
        }

    runtime = {
        "topology": payload,
        "cuts": cuts_payload,
        "out_root": str(args.out_root.resolve()),
    }

    # Optional: learned combiner head. Prefer 'topology3' because it uses exactly
    # the topology's 3 cuts (already loaded by the web app). Only wire it in if
    # its model file exists and its cuts are a subset of the loaded cuts.
    combiner_best = args.out_root / "combiner" / "topology3" / "best.json"
    if combiner_best.exists():
        cb = json.loads(combiner_best.read_text(encoding="utf-8"))
        model_path = cb.get("model_path")
        cb_cuts = list(cb.get("cuts_used", []))
        if model_path and Path(model_path).exists() and set(cb_cuts).issubset(cuts_payload):
            runtime["combiner"] = {
                "featureset": cb.get("featureset", "topology3"),
                "best_model": cb.get("best_model"),
                "model_path": str(Path(model_path).resolve()),
                "cuts": cb_cuts,  # feature column order
            }
            print(f"   combiner: {cb.get('featureset')} / {cb.get('best_model')} (cuts={cb_cuts})")
        else:
            print(f"   combiner: skipped (missing model or cuts not subset of {sorted(cuts_payload)})")

    runtime_path = args.web_app / "_runtime.json"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(json.dumps(runtime, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {runtime_path}")
    print(f"   topology: {payload['name']} {payload['description']}")
    print(f"   cuts: {sorted(cuts_payload.keys())}")


if __name__ == "__main__":
    main()
