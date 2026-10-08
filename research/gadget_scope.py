"""What counts as an electronic-gadget review.

DeFaketive's risk checker is scoped to electronic gadgets. A review is in
scope when it comes from one of our scrapes of an electronics category, or,
for sources that carry no category (FiReCS, the public Shopee set, direct
scrapes), when its text names a gadget or a gadget part.

The word test is deliberately broad: it decides which reviews are drawn for
labelling, and a person still reads every sentence.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

GADGET_CATEGORIES = frozenset({"audio", "power", "mobile_accessories", "computer_peripherals", "smart_home"})
EXCLUDED_CATEGORIES = frozenset({"out_of_scope"})
EXCLUDED_PLATFORMS = ("temu",)
GADGET = re.compile(
    r"(?<!\w)(charger|charging|charge|na ?charge|powerbank|power bank|battery|baterya|batt|lowbat\w*|lobat\w*|mah|"
    r"phone|cellphone|cp|tablet|laptop|earphone\w*|earbud\w*|headset|headphone\w*|speaker|mic|microphone|bluetooth|"
    r"wifi|usb|cable|adapter|sd ?card|memory card|flash ?drive|gb|fan|electric fan|ring ?light|ilaw|light|bulb|lamp|"
    r"flashlight|watch|smartwatch|tv|remote|mouse|keyboard|controller|camera|cctv|tripod|router|sounds?|volume|"
    r"screen|lcd|touch ?screen|saksak|plug|socket|extension|cd player|vape)(?!\w)",
    re.IGNORECASE,
)


def mentions_gadget(text: Any) -> bool:
    return bool(GADGET.search(str(text or "")))


def is_gadget(review: Mapping[str, Any]) -> bool:
    """True for a review row (as build_labelling_queue loads them) in scope."""
    category = str(review.get("category") or "").strip().casefold()
    origin = f"{review.get('platform') or ''} {review.get('source_file') or ''}".casefold()
    if category in EXCLUDED_CATEGORIES or any(name in origin for name in EXCLUDED_PLATFORMS):
        return False
    if category in GADGET_CATEGORIES:
        return True
    return mentions_gadget(review.get("review_text") or review.get("review_text_raw"))
