"""Single source of truth for the 10 binary cuts referenced by all 5 topologies.

Each cut maps the 4 raw stages (1..4) into a 2-class problem.  Class 0 ↔ the
left subset, class 1 ↔ the right subset.  This convention is what
hierarchy.predict_topology relies on."""

from __future__ import annotations

from .data import Cut

CUTS: dict[str, Cut] = {
    "1_vs_234":   Cut("1_vs_234",   (1,),       (2, 3, 4), "stage 1 vs 2/3/4"),
    "12_vs_34":   Cut("12_vs_34",   (1, 2),     (3, 4),    "stages 1+2 vs 3+4"),
    "123_vs_4":   Cut("123_vs_4",   (1, 2, 3),  (4,),      "stages 1/2/3 vs 4"),
    "2_vs_34":    Cut("2_vs_34",    (2,),       (3, 4),    "stage 2 vs 3+4 (post-1-removal)"),
    "23_vs_4":    Cut("23_vs_4",    (2, 3),     (4,),      "stages 2+3 vs 4 (post-1-removal)"),
    "1_vs_23":    Cut("1_vs_23",    (1,),       (2, 3),    "stage 1 vs 2+3 (post-4-removal)"),
    "12_vs_3":    Cut("12_vs_3",    (1, 2),     (3,),      "stages 1+2 vs 3 (post-4-removal)"),
    "2_vs_3":     Cut("2_vs_3",     (2,),       (3,),      "stage 2 vs 3"),
    "1_vs_2":     Cut("1_vs_2",     (1,),       (2,),      "stage 1 vs 2"),
    "3_vs_4":     Cut("3_vs_4",     (3,),       (4,),      "stage 3 vs 4"),
}
