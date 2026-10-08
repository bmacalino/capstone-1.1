"""Round 6 (gadget-only) labels proposed by the AI assistant.

Same method as round 5 (research/label_round5_ai.py): the assistant read each
sentence without the stratum or any checker output, applied tagging guide 1.2
and marked every answer sure or unsure; the owner decides the unsure ones and
blind-audits a sample of the sure ones. Development data, never ground truth.

The assistant first labelled slow charging as not a defect and filed heat and
"grounded" reports under safety only. On 7 October 2026 the owner ruled that
both are defects (label_round5_ai.apply_owner_rules), which is also how the
team labelled rounds 1-3. The sets below are the assistant's first answers;
the rules are applied on top when the files are written, and then the two
labels added in guide 1.4 (FIVE; see label_round5_ai.apply_five_labels).

    python research/label_round6_ai.py
"""

from __future__ import annotations

import json
from pathlib import Path

from label_round5_ai import build, final_sets

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "outputs" / "labelling_queue_round_6"
SEED = 20261007

# Row numbers (r6-00001 -> 1). Anything not listed is a sure "no".
# Row 168 ("di legit ang bilis malobat ... di man lang nakakafull charge") was
# decided as a defect while labelling but left out of the list by mistake;
# the slip was noticed and corrected the same day, before the owner's review.
SURE_YES = {
    "safety": {50, 153, 158, 167, 299, 308, 320, 337, 413, 417, 476, 485, 564, 581},
    "defect": {
        12, 17, 30, 37, 50, 56, 60, 61, 64, 83, 92, 93, 100, 103, 116, 121, 127, 140, 152, 153, 159, 168, 170, 195, 198,
        213, 223, 228, 231, 236, 237, 239, 253, 260, 263, 270, 281, 283, 287, 326, 334, 335, 336, 337, 338, 340, 343,
        347, 358, 362, 364, 370, 374, 390, 394, 395, 401, 403, 413, 414, 415, 418, 423, 433, 434, 446, 456, 461, 463,
        466, 468, 469, 474, 476, 477, 479, 485, 486, 490, 502, 505, 508, 514, 515, 517, 524, 529, 532, 542, 548, 554,
        559, 571, 574, 577, 578, 595,
    },
    "counterfeit": {157, 188, 213, 281, 284, 292, 324, 328, 334, 354, 358, 449, 505, 515, 578},
}
UNSURE_YES = {
    "safety": {88, 100, 149, 287, 358, 402, 547},
    "defect": {
        21, 32, 33, 52, 55, 71, 72, 79, 86, 87, 105, 122, 147, 150, 154, 174, 177, 186, 188, 202, 224, 243, 252, 255,
        267, 274, 280, 299, 301, 302, 321, 328, 329, 342, 383, 393, 408, 437, 441, 457, 480, 487, 539, 543, 552, 575,
        580, 583,
    },
    "counterfeit": {168, 172, 191, 237, 367, 419, 438, 474, 539, 593},
}
UNSURE_NO = {
    "safety": {491},
    "defect": {
        11, 31, 46, 75, 173, 210, 230, 249, 282, 286, 296, 319, 353, 356, 363, 378, 387, 410, 427, 450, 467, 519, 568,
        591,
    },
    "counterfeit": {56, 140, 144, 217, 300, 494, 521, 548},
}
# Slow charging with nothing else reported: a defect under the owner's rule.
SLOW_CHARGING = {135, 265, 359, 365, 489, 563, 582, 599}

# Guide 1.4: "quality" and "not_as_described", and the old answers they replace.
FIVE = {
    "sure_yes": {
        "quality": {
            21, 52, 55, 71, 105, 147, 150, 188, 195, 224, 236, 249, 252, 255, 267, 274, 321, 327, 328, 329, 383, 393,
            395, 552, 580,
        },
        "not_as_described": {29, 56, 72, 140, 182, 198, 303, 353, 367, 394, 434, 453, 464, 474, 504, 515, 556, 569, 593},
    },
    "unsure_yes": {
        "quality": {31, 75, 296, 356, 378, 387, 427, 450, 467, 591},
        "not_as_described": {62, 154, 172, 186, 442, 566},
    },
    "no_longer": {
        "defect": {
            21, 31, 52, 55, 71, 72, 75, 105, 147, 150, 154, 188, 195, 224, 236, 249, 252, 255, 267, 274, 296, 321, 328,
            329, 353, 356, 378, 383, 387, 393, 395, 410, 427, 450, 467, 552, 580, 591,
        },
        "counterfeit": {56, 140, 144, 367, 593},
    },
}


def main() -> int:
    overlap = SLOW_CHARGING & (SURE_YES["defect"] | UNSURE_YES["defect"] | UNSURE_NO["defect"])
    if overlap:
        raise SystemExit(f"slow-charging rows also labelled elsewhere: {sorted(overlap)}")
    print(json.dumps(build(FOLDER, *final_sets(SURE_YES, UNSURE_YES, UNSURE_NO, SLOW_CHARGING, FIVE), SEED)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
