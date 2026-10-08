"""Round 7 (gadget-only, weighted towards quality and not-as-described)
labels proposed by the AI assistant, with all five labels of guide 1.4.

Same method as rounds 5 and 6 (research/label_round5_ai.py): each sentence
read without the stratum or any checker output, every answer marked sure or
unsure, the owner decides the unsure ones and blind-audits a sample of the
sure ones. Development data, never ground truth. The owner's two rules of
7 October 2026 are applied when the files are written (slow charging is a
defect; a safety report is a defect as well).

    python research/label_round7_ai.py
"""

from __future__ import annotations

import json
from pathlib import Path

from label_round5_ai import build, final_sets

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "outputs" / "labelling_queue_round_7"
SEED = 20261008

# Row numbers (r7-00001 -> 1). Anything not listed is a sure "no".
SURE_YES = {
    "safety": {376, 424, 443, 452, 488},
    "defect": {
        6, 16, 23, 27, 28, 29, 35, 40, 53, 64, 69, 76, 77, 91, 93, 100, 101, 115, 116, 140, 144, 152, 164, 166, 170,
        176, 193, 202, 206, 213, 218, 224, 228, 239, 240, 250, 254, 263, 264, 267, 296, 300, 305, 315, 344, 345, 357,
        360, 361, 378, 392, 394, 399, 403, 407, 411, 412, 420, 426, 436, 451, 457, 469, 470, 471, 475, 485, 486, 488,
        492, 498, 501, 515, 518, 521, 569, 575, 578, 581, 589, 590, 597, 600,
    },
    "counterfeit": {218, 322, 469, 513, 519, 557},
}
UNSURE_YES = {
    "safety": {106, 205, 308},
    "defect": {38, 111, 128, 146, 211, 251, 257, 289, 306, 363, 372, 560, 584, 588},
    "counterfeit": {103, 128, 255, 341, 505, 597},
}
UNSURE_NO = {"safety": set(), "defect": set(), "counterfeit": {70}}
SLOW_CHARGING = {177, 570, 586}
FIVE = {
    "sure_yes": {
        "quality": {
            32, 42, 51, 56, 59, 80, 82, 91, 97, 112, 124, 125, 127, 130, 133, 142, 152, 158, 169, 181, 185, 192, 195,
            200, 203, 205, 209, 215, 218, 219, 227, 232, 239, 249, 253, 256, 266, 271, 282, 284, 299, 324, 329, 353,
            360, 394, 405, 409, 442, 452, 456, 460, 474, 487, 490, 503, 509, 514, 528, 531, 538, 561, 573,
        },
        "not_as_described": {
            34, 44, 56, 95, 104, 133, 135, 136, 161, 165, 180, 206, 259, 294, 327, 328, 341, 358, 378, 396, 404, 475,
            539, 569,
        },
    },
    "unsure_yes": {
        "quality": {4, 50, 96, 102, 117, 231, 288, 336, 350, 388, 431, 478, 493, 512, 529, 544, 564},
        "not_as_described": {70, 128, 131, 173, 355, 411, 412, 459, 593, 594, 597},
    },
    # Labelled with five labels from the start, so nothing moves.
    "no_longer": {"defect": set(), "counterfeit": set()},
}


def main() -> int:
    print(json.dumps(build(FOLDER, *final_sets(SURE_YES, UNSURE_YES, UNSURE_NO, SLOW_CHARGING, FIVE), SEED)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
