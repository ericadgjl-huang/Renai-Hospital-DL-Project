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
        winner = json.loads(winner_path.read_text(encoding="utf-8"))["decision"]
        members = [
            {
                "backbone": m["backbone"],
                "fold": m["fold"],
                "ckpt": str(Path(m["ckpt"]).resolve()),
            }
            for m in winner["members"]
        ]
        cuts_payload[cut_name] = {
            "kind": "fold_voting",
            "members": members,
            "cv_dir": str((args.out_root / "cuts" / cut_name / "cv").resolve()),
            "ensemble_dir": str((args.out_root / "cuts" / cut_name / "ensemble").resolve()),
        }

    runtime = {
        "topology": payload,
        "cuts": cuts_payload,
        "out_root": str(args.out_root.resolve()),
    }
    runtime_path = args.web_app / "_runtime.json"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(json.dumps(runtime, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {runtime_path}")
    print(f"   topology: {payload['name']} {payload['description']}")
    print(f"   cuts: {sorted(cuts_payload.keys())}")


if __name__ == "__main__":
    main()
