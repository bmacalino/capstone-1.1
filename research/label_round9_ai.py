"""Round 9 labels proposed by the AI assistant: 989 more real gadget sentences
picked for the three weak labels (research/build_round9.py).

Same method as rounds 5-8: each sentence read without its stratum or the
pointer model's scores, guide 1.4 and the owner's rules applied, every answer
marked sure or unsure. The adviser approved AI labels (PREREGISTRATION_final_v6.md,
departure 4), so no review sheet is waited for. Training only: round 9 is never
part of the held-out third or of any test.

Where the line was drawn in this round:
  not as advertised   a stated size, capacity, speed, feature or look that the
                      reviewer says is not true. "20000mah nakalagay", "sabi
                      sa description ..." with the dispute only implied are
                      marked unsure. A wrong colour or a wrong or missing item
                      is the seller's mistake and is "no", unless the reviewer
                      says the photos or the listing showed otherwise.
  poor quality        works but weak, faint, noisy, flimsy, loose, marked or
                      looks used. Weak sound from one side only is poor
                      quality; no sound from one side is a defect.
  possible fake       the reviewer says it is not the brand or not original.
                      A missing logo or "akala ko orig" is unsure.

    python research/label_round9_ai.py
"""

from __future__ import annotations

import json
from pathlib import Path

from label_round5_ai import build, final_sets

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "outputs" / "labelling_queue_round_9"
SEED = 20261009

# Row numbers (r9-00001 -> 1). Anything not listed is a sure "no".
SURE_YES = {
    "safety": set(),
    "defect": {
        1, 4, 7, 8, 9, 14, 18, 23, 31, 44, 48, 59, 63, 64, 100, 105, 121,
        132, 150, 162, 170, 171, 178, 184, 193, 201, 203, 213, 215, 223, 243, 248, 251, 257, 258, 260, 270,
        272, 273, 275, 283, 308, 314, 325, 334, 335, 336, 337, 338, 342, 352, 355, 357, 361, 368, 369, 373, 382, 383, 387,
        435, 439, 441, 445, 461, 463, 485, 492, 497, 511, 518, 527, 533, 534, 535, 544, 545, 549,
        553, 562, 564, 569, 578, 584, 605, 618, 625, 626, 639, 643, 644, 647, 666, 685, 690,
        691, 696, 706, 708, 738, 739, 754, 773, 776, 781, 786, 794, 796, 798, 803, 808, 829, 835,
        848, 849, 852, 853, 870, 873, 880, 893, 906, 920, 927, 932, 939, 941, 942, 946, 958, 970, 973, 976, 983,
    },
    "counterfeit": {30, 113, 181, 259, 437, 443, 469, 556, 565, 567, 655, 715, 952, 969, 983},
}
UNSURE_YES = {
    "safety": set(),
    "defect": {
        5, 21, 25, 131, 200, 267, 280, 290, 299, 326, 360, 433, 464, 474, 509, 523, 539, 555, 577, 588, 607, 630,
        718, 806, 821, 855, 913,
    },
    "counterfeit": {70, 129, 192, 215, 227, 257, 268, 309, 341, 351, 430, 439, 450, 551, 686, 728, 840, 865, 893, 923},
}
# "mabilis lang uminit" beside a sound complaint: mild heat, left as no.
UNSURE_NO = {"safety": {943}, "defect": set(), "counterfeit": set()}
# Slow charging was already labelled a defect while reading this round.
SLOW_CHARGING: set[int] = set()
FIVE = {
    "sure_yes": {
        "quality": {
            23, 38, 45, 48, 57, 66, 67, 68, 69, 80, 89, 90, 97, 101, 106, 116, 130,
            149, 151, 156, 165, 187, 194, 202, 211, 222, 228, 238, 246, 254, 263,
            282, 284, 286, 299, 318, 323, 349, 354, 362, 365, 366, 373, 386, 388, 394, 397, 399, 402, 406, 409, 410,
            418, 426, 427, 431, 433, 438, 442, 447, 448, 451, 454, 464, 478, 484, 493, 500, 504, 507, 531, 535, 537, 540,
            558, 560, 569, 573, 580, 585, 592, 601, 612, 613, 624, 628, 631, 633, 635, 645, 663, 664, 672, 678,
            691, 704, 712, 716, 722, 734, 747, 759, 775, 780, 783, 788, 795, 798, 799, 801, 810, 811, 819, 820, 821, 826,
            829, 832, 835, 837,
            850, 851, 852, 854, 855, 863, 868, 875, 884, 896, 897, 898, 899, 907, 909, 915, 917, 918, 922, 938, 940, 943,
            947, 949, 960, 964, 971, 974, 978, 984,
        },
        "not_as_described": {
            5, 6, 53, 54, 62, 75, 93, 98, 104, 115, 119, 125, 127,
            134, 143, 148, 195, 257, 259, 268, 269,
            271, 275, 278, 285, 291, 301, 328, 339, 350, 351, 352, 363, 376, 384, 400,
            415, 425, 430, 458, 472, 476, 480, 492, 499, 504, 505, 522, 532, 548,
            551, 553, 562, 572, 579, 583, 595, 621, 623, 662, 675, 677, 683,
            703, 705, 711, 714, 761, 770, 775, 782, 785, 790, 791, 793, 807, 818,
            848, 877, 880, 885, 886, 908, 910, 924, 950, 958, 970, 982, 987, 988,
        },
    },
    "unsure_yes": {
        "quality": {
            8, 13, 34, 37, 52, 71, 76, 99, 159, 162, 164, 231, 256, 259, 302, 331, 358, 396, 407, 413, 441, 453, 470, 471,
            502, 508, 521, 524, 538, 552, 559, 593, 606, 609, 620, 627, 629, 642, 648, 649, 668, 681, 690, 698, 731, 744,
            771, 779, 805, 816, 828, 872, 881, 888, 921, 953, 959, 980,
        },
        "not_as_described": {
            11, 14, 32, 33, 47, 49, 58, 61, 72, 73, 84, 86, 94, 95, 124, 137, 150, 163, 169, 199, 203, 204, 221, 241, 252,
            253, 264, 265, 292, 294, 315, 325, 357, 364, 372, 385, 401, 414, 444, 471, 489, 491, 510, 519, 527, 541, 544,
            547, 561, 594, 622, 637, 638, 640, 651, 658, 659, 681, 726, 735, 753, 757, 765, 766, 797, 800, 805, 843, 847,
            861, 876, 892, 894, 895, 900, 912, 929, 934, 939, 944, 949, 952, 955, 956, 962, 963,
        },
    },
    "no_longer": {"defect": set(), "counterfeit": set()},
}


def main() -> int:
    print(json.dumps(build(FOLDER, *final_sets(SURE_YES, UNSURE_YES, UNSURE_NO, SLOW_CHARGING, FIVE), SEED)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
