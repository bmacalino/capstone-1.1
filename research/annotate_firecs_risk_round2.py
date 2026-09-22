from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/firecs_risk_annotation_round_2_targeted/_work/firecs_round2_selection.json"
OUTPUT = ROOT / "outputs/firecs_risk_annotation_round_2_targeted/_work/firecs_round2_annotator_2.json"


def has(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in patterns)


WRONG_MISSING = [
    r"\bwrong item\b", r"\bmaling item\b", r"\bmali (?:ang|yung|ung) (?:item|product|pinadala|dumating)\b",
    r"\biba (?:ang|yung|ung) (?:item|product|dumating|pinadala)\b", r"\bkulang\b", r"\bmissing\b",
    r"\bwalang (?:charger|cable|cord|remote|battery|accessor|piece|piraso|kasama)\b",
    r"\border ko\b.{0,55}\b(?:pero|kaso|pinadala|dumating)\b.{0,55}\b(?:iba|oppo|iphone|fan|massage|mouse pad)\b",
    r"\bexpect ko\b.{0,35}\b(?:pero|kaso|dumating|fan|iba)\b", r"\bscam+m*er\b.{0,80}\b(?:order|dumating|item)\b",
]

FUNCTION_FAILURE = [
    r"\b(?:hindi|di|ayaw|wala(?:ng)?) (?:na )?(?:gumana|gumagana|umandar|umiilaw|mag on|mag-on|magcharge|mag charge|mag connect|kumonek|mag pair|tumunog)\b",
    r"\b(?:not|isn t|isnt|doesn t|doesnt) (?:working|work|charging|turning on|connecting|pairing|lighting|functioning)\b",
    r"\bno (?:sound|audio|power|signal|display|light|lights|connection)\b", r"\bwalang (?:tunog|sound|audio|signal|display|ilaw)\b",
    r"\b(?:sira|defect(?:ive)?|malfunction|dead on arrival|doa)\b", r"\b(?:disconnect|disconnected|putol putol|pawala wala)\b",
    r"\bone side (?:only|works|working)\b", r"\b(?:left|right) (?:side|earphone|earbud) (?:not|isn t|isnt|doesn t|doesnt)\b",
]

PERFORMANCE = [
    r"\b(?:mabilis|madali) (?:ma)?lowbat\b", r"\bmalowbat\b", r"\blowbat\b", r"\bbattery (?:drain|drains|draining|life.{0,20}(?:short|poor|bad))\b",
    r"\b(?:mahina|hina) (?:ng|ang|yung|ung)? ?(?:sound|tunog|volume|fan|hangin|signal|battery|ilaw|light|bass|mic|microphone)?\b",
    r"\b(?:malabo|blurry|blurred)\b", r"\b(?:lag|laggy|delay|delayed)\b", r"\b(?:umiinit|overheat|overheating|sobrang init)\b",
    r"\b(?:poor|bad|low) quality\b", r"\b(?:hindi|di) (?:malakas|matagal|clear|maayos)\b", r"\b(?:noise|static|garalgal)\b",
    r"\b(?:loose|maluwag)\b", r"\b(?:hard|hirap) (?:pindutin|i press|i-press|connect|pair)\b",
]

PHYSICAL_DAMAGE = [
    r"\b(?:basag|crack(?:ed)?|broken|bali|naputol|punit|dent(?:ed)?|gasgas|scratch(?:ed)?|damaged?|deformed)\b",
]

ADVERTISED_MISMATCH = [
    r"\b(?:hindi|di) pala (?:rechargeable|wireless|bluetooth|original|waterproof|fast charg|may battery)\b",
    r"\b(?:fake|counterfeit|peke)\b", r"\b(?:not|hindi|di) (?:as )?(?:advertised|described)\b",
    r"\b(?:advertised|description).{0,45}\b(?:pero|but|not|hindi|wala)\b",
]

SAFETY = [
    r"\b(?:spark|sparking|kuryente|electric shock|nasunog|burn(?:ed|ing)?|smoke|usok|sumabog|explode|leak(?:ing)?|amoy sunog)\b",
]

SETUP_ONLY = [
    r"\b(?:di|hindi) (?:ko )?(?:alam|marunong) (?:gamitin|i connect|iconnect|i-connect|isetup|i setup|i-setup)\b",
    r"\b(?:how|paano) (?:to|ko|ba) (?:use|gamitin|connect|setup|set up|pair)\b", r"\bneed.{0,15}(?:tutorial|manual|instruction)\b",
    r"\b(?:hard|hirap) (?:to )?(?:assemble|install|set up|setup)\b",
]

PACKAGING_SERVICE = [
    r"\b(?:box|pouch|bubble wrap|packaging|parcel|balot|delivery|deliver|courier|rider|seller|shop|response|responsive|shipping)\b",
]

POSITIVE_WORKING = [
    r"\b(?:gumagana|working|works well|works fine|no problem|walang sira|maayos naman|okay naman|ok naman|good condition)\b",
]

AMBIGUOUS = [
    r"\b(?:sana|baka|parang|ata|hindi ko lang alam|di ko lang alam|not sure|maybe|might)\b",
]


MANUAL_OVERRIDE_GROUPS = {
    ("yes", "Reports a wrong or missing product, variant, quantity, or included component.", "wrong_or_missing"): {
        "firecs-risk-r2-0003", "firecs-risk-r2-0005", "firecs-risk-r2-0011", "firecs-risk-r2-0013",
        "firecs-risk-r2-0026", "firecs-risk-r2-0039", "firecs-risk-r2-0043", "firecs-risk-r2-0045",
        "firecs-risk-r2-0064", "firecs-risk-r2-0068", "firecs-risk-r2-0073", "firecs-risk-r2-0084",
        "firecs-risk-r2-0088", "firecs-risk-r2-0111", "firecs-risk-r2-0128", "firecs-risk-r2-0133",
        "firecs-risk-r2-0169", "firecs-risk-r2-0174", "firecs-risk-r2-0186", "firecs-risk-r2-0206",
        "firecs-risk-r2-0221", "firecs-risk-r2-0227", "firecs-risk-r2-0235", "firecs-risk-r2-0241",
        "firecs-risk-r2-0247", "firecs-risk-r2-0253", "firecs-risk-r2-0270", "firecs-risk-r2-0277",
        "firecs-risk-r2-0301", "firecs-risk-r2-0303", "firecs-risk-r2-0307", "firecs-risk-r2-0322",
        "firecs-risk-r2-0328", "firecs-risk-r2-0348", "firecs-risk-r2-0349", "firecs-risk-r2-0352",
        "firecs-risk-r2-0366", "firecs-risk-r2-0380", "firecs-risk-r2-0386", "firecs-risk-r2-0387",
    },
    ("yes", "Reports a product malfunction or failed function.", "function_failure"): {
        "firecs-risk-r2-0001", "firecs-risk-r2-0024", "firecs-risk-r2-0025", "firecs-risk-r2-0028",
        "firecs-risk-r2-0035", "firecs-risk-r2-0066", "firecs-risk-r2-0070", "firecs-risk-r2-0079",
        "firecs-risk-r2-0085", "firecs-risk-r2-0110", "firecs-risk-r2-0121", "firecs-risk-r2-0144",
        "firecs-risk-r2-0158", "firecs-risk-r2-0163", "firecs-risk-r2-0165", "firecs-risk-r2-0178",
        "firecs-risk-r2-0208", "firecs-risk-r2-0211", "firecs-risk-r2-0217", "firecs-risk-r2-0224",
        "firecs-risk-r2-0226", "firecs-risk-r2-0229", "firecs-risk-r2-0231", "firecs-risk-r2-0244", "firecs-risk-r2-0246",
        "firecs-risk-r2-0259", "firecs-risk-r2-0289", "firecs-risk-r2-0290", "firecs-risk-r2-0293",
        "firecs-risk-r2-0294", "firecs-risk-r2-0314", "firecs-risk-r2-0332", "firecs-risk-r2-0334",
        "firecs-risk-r2-0337", "firecs-risk-r2-0343", "firecs-risk-r2-0346", "firecs-risk-r2-0378",
        "firecs-risk-r2-0388", "firecs-risk-r2-0394",
    },
    ("yes", "Reports weak, unreliable, or degraded product performance.", "performance"): {
        "firecs-risk-r2-0009", "firecs-risk-r2-0017", "firecs-risk-r2-0089", "firecs-risk-r2-0101",
        "firecs-risk-r2-0102", "firecs-risk-r2-0120", "firecs-risk-r2-0125", "firecs-risk-r2-0143",
        "firecs-risk-r2-0162", "firecs-risk-r2-0164", "firecs-risk-r2-0173", "firecs-risk-r2-0183",
        "firecs-risk-r2-0202",
        "firecs-risk-r2-0228", "firecs-risk-r2-0232", "firecs-risk-r2-0238", "firecs-risk-r2-0304",
        "firecs-risk-r2-0155", "firecs-risk-r2-0281", "firecs-risk-r2-0308",
        "firecs-risk-r2-0340", "firecs-risk-r2-0365", "firecs-risk-r2-0393",
    },
    ("yes", "Reports multiple product issues with an earlier unit.", "reported_prior_issue"): {
        "firecs-risk-r2-0251",
    },
    ("yes", "Reports a mismatch with the advertised product or supplied condition.", "advertised_mismatch"): {
        "firecs-risk-r2-0008", "firecs-risk-r2-0338",
    },
    ("yes", "Reports physical damage or a product defect.", "physical_defect"): {
        "firecs-risk-r2-0126", "firecs-risk-r2-0129", "firecs-risk-r2-0131", "firecs-risk-r2-0160",
        "firecs-risk-r2-0192", "firecs-risk-r2-0195", "firecs-risk-r2-0278", "firecs-risk-r2-0286",
        "firecs-risk-r2-0321", "firecs-risk-r2-0347", "firecs-risk-r2-0359",
    },
    ("no", "No confirmed product issue; the comment is an expectation, preference, or mystery-item outcome.", "expectation_or_preference"): {
        "firecs-risk-r2-0004", "firecs-risk-r2-0012", "firecs-risk-r2-0018", "firecs-risk-r2-0051",
        "firecs-risk-r2-0137",
        "firecs-risk-r2-0171", "firecs-risk-r2-0188", "firecs-risk-r2-0216", "firecs-risk-r2-0218",
        "firecs-risk-r2-0220", "firecs-risk-r2-0268", "firecs-risk-r2-0287", "firecs-risk-r2-0302",
        "firecs-risk-r2-0309", "firecs-risk-r2-0341", "firecs-risk-r2-0342", "firecs-risk-r2-0351",
        "firecs-risk-r2-0361", "firecs-risk-r2-0376", "firecs-risk-r2-0383", "firecs-risk-r2-0396",
    },
    ("no", "The review reports acceptable operation and no actual product failure.", "working_no_issue"): {
        "firecs-risk-r2-0022", "firecs-risk-r2-0037", "firecs-risk-r2-0049", "firecs-risk-r2-0067",
        "firecs-risk-r2-0081",
        "firecs-risk-r2-0083", "firecs-risk-r2-0093", "firecs-risk-r2-0135", "firecs-risk-r2-0201",
        "firecs-risk-r2-0265", "firecs-risk-r2-0266", "firecs-risk-r2-0274", "firecs-risk-r2-0297",
        "firecs-risk-r2-0319", "firecs-risk-r2-0323", "firecs-risk-r2-0360", "firecs-risk-r2-0390",
        "firecs-risk-r2-0399",
    },
    ("no", "This record is outside the electronics product-risk scope and gives no usable device-risk evidence.", "out_of_scope"): {
        "firecs-risk-r2-0119", "firecs-risk-r2-0367", "firecs-risk-r2-0391",
    },
    ("no", "Complaint concerns delivery or packaging; the product itself is reported as working.", "service_or_packaging"): {
        "firecs-risk-r2-0094",
    },
    ("no", "The complaint is too vague to establish an actual product risk issue.", "vague_no_risk"): {
        "firecs-risk-r2-0034",
    },
}

MANUAL_OVERRIDES = {
    review_id: result
    for result, review_ids in MANUAL_OVERRIDE_GROUPS.items()
    for review_id in review_ids
}


def classify(record: dict) -> tuple[str, str, str]:
    if record["review_id"] in MANUAL_OVERRIDES:
        return MANUAL_OVERRIDES[record["review_id"]]

    text = record["review_text"].lower().strip()
    sentiment = record["firecs_sentiment"]
    probability = float(record["model_risk_probability"])

    safety = has(text, SAFETY)
    wrong_missing = has(text, WRONG_MISSING)
    function_failure = has(text, FUNCTION_FAILURE)
    performance = has(text, PERFORMANCE)
    physical = has(text, PHYSICAL_DAMAGE)
    mismatch = has(text, ADVERTISED_MISMATCH)
    setup_only = has(text, SETUP_ONLY)
    service = has(text, PACKAGING_SERVICE)
    working = has(text, POSITIVE_WORKING)
    ambiguous = has(text, AMBIGUOUS)

    # Packaging damage alone is not a product defect.
    package_damage_only = physical and service and has(text, [r"\b(?:box|pouch|parcel|packaging|balot)\b"]) and not has(
        text, [r"\b(?:item|product|unit|earphone|charger|cable|mouse|keyboard|fan|remote|device)\b.{0,20}\b(?:basag|crack|broken|bali|naputol|damaged?)\b"]
    )

    if safety:
        return "yes", "Reports a product safety hazard.", "safety"
    if wrong_missing:
        return "yes", "Reports a wrong or missing product/component.", "wrong_or_missing"
    if physical and not package_damage_only:
        return "yes", "Reports physical damage or a product defect.", "physical_defect"
    if mismatch:
        return "yes", "Reports a mismatch with an advertised product feature.", "advertised_mismatch"
    if function_failure:
        if ambiguous and not has(text, [r"\b(?:hindi|di|ayaw|wala(?:ng)?|not|doesn t|doesnt|sira|defect)\b"]):
            return "unsure", "Possible product failure, but the review does not confirm it.", "ambiguous_failure"
        return "yes", "Reports a product malfunction or failed function.", "function_failure"
    if performance:
        return "yes", "Reports weak, unreliable, or degraded product performance.", "performance"
    if setup_only:
        return "no", "Setup or usage difficulty without evidence that the product failed.", "setup_only"
    if package_damage_only or (service and sentiment == "negative" and probability < 0.70):
        return "no", "Complaint concerns packaging, delivery, or seller service rather than the product.", "service_or_packaging"
    if working:
        return "no", "The review says the product works and gives no confirmed product issue.", "working_no_issue"

    # Reviews near the sampling boundary are deliberately retained as unsure when
    # the text lacks enough detail for a defensible product-only decision.
    if ambiguous and sentiment != "positive":
        return "unsure", "The review suggests a possible issue but does not confirm a product failure.", "ambiguous"
    if 0.40 <= probability <= 0.54 and sentiment != "positive":
        return "unsure", "Insufficient detail to determine whether the product itself has a risk issue.", "boundary_unclear"

    if sentiment == "positive":
        return "no", "Positive feedback with no actual product issue reported.", "positive_no_issue"
    if sentiment == "neutral" and probability < 0.40:
        return "no", "No actual product malfunction, defect, or performance failure is reported.", "neutral_no_issue"
    if sentiment == "negative" and service:
        return "no", "Negative feedback is about service or delivery, not an actual product issue.", "service_only"
    if sentiment == "negative" and probability >= 0.70:
        return "unsure", "Negative review, but the product issue is not specific enough to confirm risk.", "negative_unclear"

    return "no", "No actual product malfunction, defect, or performance failure is reported.", "no_product_issue"


def main() -> None:
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    annotations = []
    for record in payload["records"]:
        risk, note, reason = classify(record)
        annotations.append(
            {
                "review_id": record["review_id"],
                "risk": risk,
                "notes": note,
                "reason": reason,
                "review_text": record["review_text"],
                "firecs_sentiment": record["firecs_sentiment"],
                "model_risk_probability": record["model_risk_probability"],
            }
        )

    counts: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for annotation in annotations:
        counts[annotation["risk"]] = counts.get(annotation["risk"], 0) + 1
        reasons[annotation["reason"]] = reasons.get(annotation["reason"], 0) + 1

    output = {"annotator_id": "Codex", "counts": counts, "reason_counts": reasons, "annotations": annotations}
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": counts, "reason_counts": reasons}, indent=2))


if __name__ == "__main__":
    main()
