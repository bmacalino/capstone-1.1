"""Score scratch versions of the rule file against the served one.

Two configurations, both fixed before the held-out third was scored:
  A  only patterns that found a right report in sentences they were not
     written from (extra_labels_r3.py)
  B  A plus every round-9 pattern that was right at least twice where it was
     written and added no wrong flag anywhere (precision confirmed on other
     sentences, recall not confirmed)
Rule for both: held-out F1 at least 0.02 above the served rules and at least
70 in 100 of the added flags right. If both pass, B.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

import candidates as c

HERE = Path(__file__).parent


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_b():
    results = json.loads((HERE / "r3_candidates.json").read_text(encoding="utf-8"))
    wide = {"not_as_described": [], "quality": []}
    for row in results:
        if row["name"].startswith("r2_") or row["keep"]:
            continue
        if row["source"][0] >= 2 and row["source"][1] == 0 and row["check"][1] == 0:
            source = c.NAD_R9 if row["label"] == "not_as_described" else c.QUALITY_R9
            wide[row["label"]].append((row["name"], source[row["name"]]))
    text = (HERE / "extra_labels_r3.py").read_text(encoding="utf-8")
    for label, anchor in (
        ("not_as_described", '        r"(?:charger|fan|mic|speaker|mouse)(?!\\w)",\n'),
        ("quality", '        r"(?:sobrang|subrang|masyadong)\\s+gaan(?!\\w)",\n'),
    ):
        assert text.count(anchor) == 1, anchor
        extra = "".join(f"        {pattern!r},  # {name}\n" for name, pattern in wide[label])
        text = text.replace(anchor, anchor + extra)
    (HERE / "extra_labels_r3_wide.py").write_text(text, encoding="utf-8", newline="\n")
    return {label: [name for name, _ in rows] for label, rows in wide.items()}


def main():
    print("B adds:", json.dumps(build_b()))
    modules = {"served": c.current, "A": load(HERE / "extra_labels_r3.py"), "B": load(HERE / "extra_labels_r3_wide.py")}
    gadget = np.array([bool(r.get("gadget")) for r in c.records])
    sets = [("round 9", c.ROUND9), ("dev rounds 1-8", c.DEV), ("HELD-OUT all", c.held), ("HELD-OUT gadget only", c.held & gadget)]
    for label, function in (("not_as_described", "not_as_described_spans"), ("quality", "quality_spans")):
        truth = c.truth_of(label)
        labelled = np.array([r["labels"].get(label) is not None for r in c.records])
        flags = {name: np.array([bool(getattr(m, function)(t)) for t in c.norm]) for name, m in modules.items()}
        print(label)
        for set_name, mask in sets:
            mask = mask & labelled
            line = f"  {set_name:22} reports {int((truth & mask).sum()):4}"
            base_f1 = None
            for name, f in flags.items():
                tp, fp, fn = int((f & truth & mask).sum()), int((f & ~truth & mask).sum()), int((~f & truth & mask).sum())
                p = tp / (tp + fp) if tp + fp else 0.0
                r = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * p * r / (p + r) if p + r else 0.0
                add = f & ~flags["served"] & mask
                lost = ~f & flags["served"] & mask
                extra = "" if name == "served" else f" [+{int((add & truth).sum())} right, +{int((add & ~truth).sum())} wrong, lost {int(lost.sum())}; F1 {f1 - base_f1:+.3f}]"
                base_f1 = f1 if name == "served" else base_f1
                line += f" | {name} {tp}/{fp} F1 {f1:.3f}{extra}"
            print(line)


if __name__ == "__main__":
    main()
