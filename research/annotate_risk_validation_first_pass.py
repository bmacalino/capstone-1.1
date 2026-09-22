"""Write the reviewed AI first pass for the blind risk validation queue."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "risk_development_round_1" / "risk_validation_blind_queue.csv"
OUTPUT = ROOT / "outputs" / "risk_development_round_1" / "risk_validation_ai_first_pass.csv"

# review_id: (risk, categories, severity, reason)
LABELS = {
    "lazada-r1-0004": ("yes", "defect", "3", "Unit does not work."),
    "lazada-r1-0010": ("no", "", "", "Positive review with no concrete problem."),
    "lazada-r1-0013": ("yes", "damage|delivery", "2", "Phone-holder component arrived broken after poor delivery handling."),
    "lazada-r1-0015": ("no", "", "", "Positive review with no concrete problem."),
    "lazada-r1-0017": ("yes", "defect|quality", "2", "Product became damaged after one week of use."),
    "lazada-r1-0019": ("yes", "performance|seller_service", "2", "Incorrect instructions made setup difficult and the seller did not help."),
    "lazada-r1-0020": ("no", "", "", "Satisfied review with no concrete problem."),
    "lazada-r1-0021": ("yes", "performance", "2", "App does not work with the lock and Bluetooth pairing fails."),
    "lazada-r1-0022": ("yes", "quality|seller_service", "2", "Plastic construction contradicts the misleading metal-like listing photo."),
    "lazada-r1-0023": ("yes", "performance", "1", "Handle operation is reversed when mounted on the right."),
    "lazada-r1-0024": ("yes", "defect", "3", "Ring light emitted smoke on its second use."),
    "lazada-r1-0026": ("yes", "quality|seller_service", "2", "Product is low quality and does not match its description."),
    "lazada-r1-0028": ("yes", "performance", "2", "Stand does not fit the ring light."),
    "lazada-r1-0040": ("no", "", "", "Positive review with no concrete problem."),
    "lazada-r1-0042": ("yes", "performance", "1", "Maximum volume is insufficient."),
    "lazada-r1-0044": ("yes", "defect|seller_service", "3", "Doorbell and monitor cannot pair and the item appears used or refurbished."),
    "lazada-r1-0048": ("yes", "defect", "3", "Doorbell camera never worked and could not pair with the display."),
    "lazada-r1-0051": ("yes", "performance", "1", "Camera and voice work only at short range."),
    "lazada-r1-0062": ("yes", "performance", "2", "Battery drains rapidly and the charging hardware performs poorly."),
    "lazada-r1-0071": ("yes", "performance", "2", "Battery drains easily and device charging fails without a cable."),
    "lazada-r1-0095": ("no", "", "", "Positive review with no concrete problem."),
    "lazada-r1-0100": ("no", "", "", "Positive feature review with no concrete problem."),
    "lazada-r1-0104": ("no", "", "", "Complaint concerns posting on Lazada and gives no product risk evidence."),
    "lazada-r1-0105": ("no", "", "", "Positive review with no concrete problem."),
    "lazada-r1-0108": ("no", "", "", "Positive review with no concrete problem."),
    "lazada-r1-0117": ("yes", "wrong_item|seller_service", "3", "Received a basic knob without advertised Bluetooth or Wi-Fi and seller did not respond."),
    "lazada-r1-0118": ("yes", "performance", "1", "Tuya app performance is described as poor."),
    "lazada-r1-0119": ("yes", "performance", "2", "Setup is difficult, auto-lock fails, and the app cannot control the lock."),
    "lazada-r1-0122": ("yes", "quality|performance|wrong_item", "2", "Manual and remote battery are missing, mounting is unstable, and switch labels are reversed."),
    "lazada-r1-0123": ("no", "", "", "Item and delivery are acceptable; remaining text is a general listing suggestion."),
    "lazada-r1-0124": ("yes", "wrong_item|quality|seller_service", "3", "Received a cheaply made product materially different from the advertised photos."),
    "lazada-r1-0125": ("no", "", "", "Positive feature review with no concrete problem."),
    "lazada-r1-0126": ("yes", "performance", "2", "Advertised remote Wi-Fi access cannot be used."),
    "lazada-r1-0131": ("yes", "defect|quality", "2", "Push button does not work and material is below expectation."),
    "lazada-r1-0142": ("yes", "performance", "3", "Rapid battery drain makes the monitor unusable."),
    "lazada-r1-0143": ("yes", "performance", "1", "Sound is barely audible from the second floor."),
    "lazada-r1-0144": ("yes", "defect", "3", "Defective unit cannot pair and is described as not working."),
    "lazada-r1-0156": ("yes", "performance", "1", "Microphone is delayed and requires repeated activation."),
    "lazada-r1-0157": ("yes", "defect", "2", "Headphones work but their charging port does not."),
    "lazada-r1-0176": ("yes", "defect", "3", "Device charges but no longer operates."),
    "lazada-r1-0180": ("no", "", "", "Positive feature review with no concrete problem."),
    "lazada-r1-0185": ("no", "", "", "Positive feature review with no concrete problem."),
    "lazada-r1-0190": ("yes", "wrong_item", "3", "Required component connecting the stand and ring light is missing."),
    "lazada-r1-0197": ("yes", "defect", "3", "Lock can be opened without authentication, defeating its security function."),
    "lazada-r1-0199": ("yes", "performance", "2", "Phone heats easily during gaming even on Wi-Fi."),
    "lazada-r1-0203": ("no", "", "", "Positive feature review with no concrete problem."),
    "lazada-r1-0210": ("no", "", "", "Positive review with no concrete problem."),
    "lazada-r1-0213": ("yes", "damage", "2", "Item has a cracked edge despite remaining usable."),
    "lazada-r1-0214": ("yes", "seller_service", "2", "Return option became unavailable after the buyer was induced to receive the item."),
    "lazada-r1-0217": ("yes", "performance", "2", "Mounting and remote security design make the product unsuitable for a gate."),
    "lazada-r1-0219": ("yes", "wrong_item|seller_service", "2", "Received product does not match the listing photo."),
    "lazada-r1-0232": ("yes", "defect", "3", "Earbuds do not charge and therefore cannot connect."),
    "lazada-r1-0233": ("yes", "performance", "2", "Product cannot be installed through its app."),
    "lazada-r1-0239": ("yes", "performance", "2", "Device cannot be added after following the manual and the button is difficult to press."),
    "lazada-r1-0241": ("yes", "seller_service", "2", "Listing falsely advertises the product as waterproof."),
    "lazada-r1-0261": ("no", "", "", "Sound quality is acceptable."),
    "lazada-r1-0265": ("no", "", "", "Positive feature review with no concrete problem."),
    "lazada-r1-0270": ("no", "", "", "Positive installation review with no concrete problem."),
    "lazada-r1-0275": ("no", "", "", "Satisfied review with no concrete problem."),
    "lazada-r1-0277": ("yes", "performance", "2", "Monitor battery drains quickly."),
    "lazada-r1-0283": ("no", "", "", "Positive feature review with no concrete problem."),
    "lazada-r1-0287": ("yes", "defect", "3", "Right earpiece is defective and the headset cannot be used normally."),
    "lazada-r1-0294": ("yes", "seller_service", "2", "Buyer reports misleading pricing and unaccommodating customer service."),
    "lazada-r1-0295": ("yes", "performance", "1", "Doorbell rings repeatedly after one press."),
    "lazada-r1-0309": ("yes", "performance", "1", "Product is difficult to install because qualified installation support is unavailable."),
    "lazada-r1-0314": ("yes", "seller_service", "1", "Seller cannot answer the buyer's question."),
    "lazada-r1-0326": ("yes", "performance", "1", "Video is unclear compared with similar cameras but remains usable."),
    "lazada-r1-0332": ("yes", "performance", "2", "Camera is blurry and does not deliver advertised HD output."),
    "lazada-r1-0334": ("yes", "defect|performance", "3", "Replacement failed after days and has severe battery and charging problems."),
    "lazada-r1-0342": ("yes", "defect", "3", "Product stopped working and no longer charges."),
    "lazada-r1-0346": ("no", "", "", "Initial pairing difficulty was resolved and the product now works acceptably."),
    "lazada-r1-0350": ("no", "", "", "Positive review with no concrete problem."),
    "lazada-r1-0355": ("no", "", "", "Positive product description with no concrete problem."),
    "lazada-r1-0360": ("no", "", "", "Positive review; needing a USB adapter is not described as a failure."),
    "lazada-r1-0379": ("yes", "delivery", "2", "Parcel was forgotten or left at another location."),
    "lazada-r1-0380": ("no", "", "", "Item arrived intact and as expected."),
    "lazada-r1-0389": ("yes", "defect|quality", "3", "Lock does not work because internal wires are broken."),
    "lazada-r1-0394": ("yes", "delivery", "3", "Order was never received."),
    "lazada-r1-0411": ("yes", "performance|quality", "2", "Night video is unclear and the receiver display requires tape to stay mounted."),
    "lazada-r1-0426": ("no", "", "", "Sound quality is good."),
    "lazada-r1-0435": ("no", "", "", "Durability rating is positive and gives no concrete problem."),
    "lazada-r1-0440": ("no", "", "", "Positive design and material review with no concrete problem."),
    "lazada-r1-0445": ("no", "", "", "Positive feature review with no concrete problem."),
    "lazada-r1-0449": ("no", "", "", "Positive fragment and clipboard filler contain no concrete risk evidence."),
    "lazada-r1-0459": ("no", "", "", "Empty quality field contains no risk evidence."),
    "lazada-r1-0464": ("yes", "wrong_item|performance", "2", "Promised bag is missing and ring-light output is weak."),
    "lazada-r1-0465": ("no", "", "", "Positive feature review with no concrete problem."),
    "lazada-r1-0486": ("yes", "performance", "2", "Installation is difficult and battery discharges rapidly."),
    "lazada-r1-0501": ("no", "", "", "Average sound alone does not identify a concrete failure."),
    "lazada-r1-0504": ("yes", "wrong_item", "1", "Received orientation does not match the buyer's required handedness and cannot be converted."),
    "lazada-r1-0520": ("no", "", "", "Positive review with no concrete problem."),
    "lazada-r1-0525": ("no", "", "", "Positive feature review with no concrete problem."),
    "lazada-r1-0530": ("no", "", "", "Item is good, well packed, and delivered quickly."),
    "lazada-r1-0544": ("yes", "defect|performance|seller_service", "3", "Smart-lock authentication and app features fail and seller provides no useful help."),
    "lazada-r1-0550": ("no", "", "", "Positive feature review with no concrete problem."),
    "lazada-r1-0556": ("yes", "wrong_item", "3", "Paid monitor component is missing from the order."),
    "lazada-r1-0571": ("yes", "performance", "1", "Sound quality is poor."),
    "lazada-r1-0584": ("yes", "defect", "3", "Auto-lock falsely reports success while leaving the door unlocked."),
    "lazada-r1-0605": ("no", "", "", "Positive feature review with no concrete problem."),
    "lazada-r1-0610": ("no", "", "", "Placeholder text contains no product-risk claim."),
    "lazada-r1-0615": ("no", "", "", "Positive feature fragment with no concrete problem."),
    "lazada-r1-0624": ("yes", "wrong_item|seller_service", "3", "Wi-Fi and battery are missing and the seller does not respond."),
    "lazada-r1-0631": ("no", "", "", "Battery rating is acceptable and gives no concrete failure."),
    "lazada-r1-0635": ("no", "", "", "Positive feature review with no concrete problem."),
    "lazada-r1-0662": ("yes", "performance", "1", "Missing manual makes setup difficult."),
    "lazada-r1-0680": ("no", "", "", "Sound quality is acceptable."),
    "lazada-r1-0683": ("no", "", "", "Positive compatibility review with no concrete problem."),
    "lazada-r1-0687": ("no", "", "", "Positive feature review with no concrete problem."),
    "lazada-r1-0691": ("no", "", "", "Product is easy to operate and fits the buyer's phone."),
    "lazada-r1-0707": ("yes", "performance", "1", "Product works but mobile connection cannot be established or found."),
    "lazada-r1-0720": ("yes", "performance", "1", "Right earbud output is weaker than the other side."),
    "lazada-r1-0747": ("no", "", "", "Positive feature review with no concrete problem."),
    "lazada-r1-0751": ("no", "", "", "Positive feature review with no concrete problem."),
    "lazada-r1-0755": ("no", "", "", "Positive feature review with no concrete problem."),
    "lazada-r1-0760": ("yes", "performance", "2", "Battery charges slowly and drains rapidly."),
    "lazada-r1-0771": ("no", "", "", "Initial power-supply misunderstanding was resolved and the product works."),
    "lazada-r1-0796": ("no", "", "", "Sound is acceptable for the price."),
    "lazada-r1-0811": ("yes", "performance", "1", "Stand is too high for the buyer's smaller laptop."),
    "lazada-r1-0815": ("no", "", "", "Positive resolved review with no active problem."),
    "lazada-r1-0819": ("no", "", "", "Positive feature review with no concrete problem."),
    "lazada-r1-0832": ("yes", "defect", "3", "Product is defective and does not charge."),
    "lazada-r1-0835": ("yes", "performance", "1", "Pairing required three hours despite eventually succeeding."),
    "lazada-r1-0868": ("no", "", "", "Positive feature review with no concrete problem."),
    "lazada-r1-0875": ("no", "", "", "Item was received in good condition; trailing fragment gives no problem."),
    "lazada-r1-0879": ("no", "", "", "Positive repeat-purchase review with no concrete problem."),
    "lazada-r1-0883": ("no", "", "", "Positive lighting review with no concrete problem."),
    "lazada-r1-0899": ("no", "", "", "Positive feature review with no concrete problem."),
    "lazada-r1-0904": ("yes", "delivery", "1", "Item arrived without a box and with only plastic packaging."),
    "lazada-r1-0933": ("no", "", "", "Average battery rating gives no concrete failure."),
    "lazada-r1-0941": ("yes", "defect", "2", "Included charging cable does not work."),
    "lazada-r1-0949": ("yes", "performance", "2", "Earbuds cannot be used properly because they are too large to fit."),
    "lazada-r1-0957": ("no", "", "", "Positive feature review with no concrete problem."),
    "lazada-r1-0964": ("no", "", "", "Battery life is described as good."),
    "lazada-r1-0970": ("yes", "defect", "2", "One earbud does not work."),
    "lazada-r1-0976": ("yes", "performance|delivery", "2", "Audio is weak, battery drains quickly, and packaging is inadequate."),
    "lazada-r1-0982": ("yes", "performance", "2", "Audio has static, dropouts, and long delay even near the phone."),
    "lazada-r1-0988": ("yes", "damage|delivery", "2", "Case arrived cracked."),
    "lazada-r1-0994": ("yes", "performance", "1", "Muddy, overpowering bass and mild discomfort reduce performance."),
}


def main() -> None:
    with INPUT.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    queue_ids = {row["review_id"] for row in rows}
    if queue_ids != set(LABELS):
        missing = sorted(queue_ids.difference(LABELS))
        extra = sorted(set(LABELS).difference(queue_ids))
        raise ValueError(f"Annotation mapping mismatch; missing={missing}, extra={extra}")

    for row in rows:
        risk, category, severity, reason = LABELS[row["review_id"]]
        row["human_risk_detected"] = risk
        row["human_risk_category"] = category
        row["human_risk_severity"] = severity
        row["decision_reason"] = reason
        row["status"] = "AI_First_Pass"

    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    risk_counts = Counter(row["human_risk_detected"] for row in rows)
    severity_counts = Counter(
        row["human_risk_severity"] for row in rows if row["human_risk_detected"] == "yes"
    )
    category_counts = Counter(
        category
        for row in rows
        for category in row["human_risk_category"].split("|")
        if category
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "risk": dict(sorted(risk_counts.items())),
                "severity": dict(sorted(severity_counts.items())),
                "categories": dict(sorted(category_counts.items())),
                "status": "AI_First_Pass",
                "output": str(OUTPUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
