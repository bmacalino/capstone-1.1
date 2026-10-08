"""Round 8 labels proposed by the AI assistant: sentences the defect models
flag in ordinary, unused gadget reviews (research/build_ordinary_round8.py).

Same method as rounds 5-7: each sentence read without its stratum or which
model flagged it, guide 1.4 and the owner's rules applied (slow charging is a
defect; a safety report is a defect as well), every answer marked sure or
unsure. Development data, never ground truth. The live_check part of this
round is never used for training (research/five_label_data.py).

    python research/label_round8_ai.py
"""

from __future__ import annotations

import json
from pathlib import Path

from label_round5_ai import build, final_sets

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "outputs" / "labelling_queue_round_8"
SEED = 20261010

# Row numbers (r8-00001 -> 1). Anything not listed is a sure "no".
SURE_YES = {
    "safety": {231, 499},
    "defect": {
        6, 9, 10, 12, 13, 20, 24, 25, 29, 30, 31, 37, 38, 43, 45, 47, 48, 49, 51, 58, 60, 61, 62, 63, 65, 66, 67, 68,
        71, 72, 73, 79, 80, 81, 82, 83, 84, 85, 89, 90, 91, 92, 96, 100, 102, 103, 104, 105, 106, 108, 109, 111, 112,
        116, 117, 118, 121, 124, 127, 128, 129, 130, 133, 134, 136, 138, 140, 141, 145, 147, 150, 152, 155, 158, 159,
        160, 161, 169, 171, 173, 175, 177, 181, 182, 186, 187, 188, 193, 195, 197, 199, 203, 205, 209, 211, 212, 213,
        219, 220, 221, 223, 226, 227, 228, 229, 230, 231, 233, 235, 236, 237, 242, 243, 245, 247, 248, 249, 250, 251,
        253, 255, 256, 260, 263, 270, 272, 275, 277, 279, 282, 285, 288, 289, 291, 293, 295, 301, 302, 303, 307, 309,
        310, 311, 316, 318, 319, 322, 325, 326, 327, 330,
        332, 333, 335, 338, 340, 342, 343, 346, 347, 349, 351, 354, 355, 357, 360, 363, 364, 371, 372, 375, 385, 386,
        387, 389, 393, 398, 399, 401, 404, 405, 406, 408, 409, 410, 414, 415, 426, 430, 431, 432, 433, 434, 435, 437,
        440, 443, 444, 446, 453, 455, 456, 458, 463, 464, 468, 469, 470, 472, 473, 475, 479, 480, 481, 486, 488, 491,
        492, 493, 495, 499, 501, 502, 503, 506, 507, 508, 510, 512, 513, 514, 517, 518, 519, 521, 530, 531, 532, 533,
        534, 538, 541, 542, 543, 544, 546, 549, 554, 555, 557, 558, 561, 565, 566, 567, 569, 571, 573, 578, 579, 580,
        583, 586, 588, 589, 590, 591, 593, 598, 599, 602, 605, 606, 609, 610, 612, 613, 614, 616, 620, 622, 626, 627,
        630, 633, 640, 641, 642, 645, 646, 647, 650, 651, 653, 656, 657,
        661, 668, 669, 675, 678, 680, 684, 686, 691, 692, 697, 701, 703, 704, 706, 707, 708, 709, 710, 713, 716, 719,
        720, 722, 726, 727, 731, 732, 737, 739, 740, 742, 743, 744, 749,
    },
    "counterfeit": set(),
}
UNSURE_YES = {
    "safety": set(),
    "defect": {
        3, 4, 23, 34, 40, 94, 132, 166, 172, 180, 185, 201, 225, 246, 266,
        341, 359, 396, 402, 403, 417, 420, 421, 424, 439, 476, 477, 485, 522, 528, 537, 547, 556, 575, 577, 595, 608,
        624, 649, 662, 667, 689, 730, 750,
    },
    "counterfeit": set(),
}
UNSURE_NO = {"safety": set(), "defect": {14, 50, 120, 142, 157}, "counterfeit": set()}
# Slow charging was already labelled a defect while reading this round.
SLOW_CHARGING: set[int] = set()
FIVE = {
    "sure_yes": {
        "quality": {
            8, 53, 76, 84, 110, 117, 161, 196, 202, 232, 238, 258, 272, 286, 308, 315, 328,
            381, 419, 438, 465, 468, 524, 567, 603, 636, 637, 691, 726, 735,
        },
        "not_as_described": {
            1, 14, 31, 44, 50, 63, 64, 101, 102, 103, 124, 151, 157, 164, 178, 226, 275, 292, 299,
            336, 344, 372, 416, 449, 476, 478, 516, 519, 609, 617, 650, 657, 664, 729,
        },
    },
    "unsure_yes": {
        "quality": {154, 215, 298, 306, 356, 448, 702, 751},
        "not_as_described": {35, 56, 99, 248, 366, 408, 436, 552, 568},
    },
    "no_longer": {"defect": set(), "counterfeit": set()},
}


def main() -> int:
    print(json.dumps(build(FOLDER, *final_sets(SURE_YES, UNSURE_YES, UNSURE_NO, SLOW_CHARGING, FIVE), SEED)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
