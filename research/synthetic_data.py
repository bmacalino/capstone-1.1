"""Training sentences written by the AI assistant (synthetic data).

The adviser approved AI labels and synthetic data (reported by the owner, 8
October 2026). Real reports of the rare problems are scarce: 112 safety and
150 fake sentences in everything labelled. These sentences are written to
teach more ways of saying each problem, and more harmless uses of the same
words.

Rules for their use, fixed here:
  * TRAINING ONLY. They are never part of a test, a left-out round or any
    figure. Models trained with them are scored on real reviews only.
  * They are not reviews. Nobody wrote them about a product they bought, and
    the app never shows them.
  * They are kept only where a model trained with them beats the served one
    on real reviews (research/train_with_synthetic.py).

How they are made. For each label the assistant wrote short core phrases in
the English, Tagalog and Taglish of marketplace reviews: reports
("positive") and look-alikes that are not reports ("negative": denials,
praise, questions, hopes, idioms, the reviewer's own other device). Each core
is turned into three sentences by adding a product word and, sometimes, an
opening or closing remark. A seed makes the file reproducible.

Second set (v2, 8 October 2026), for the three labels with the lowest F1:
poor quality, possible fake, not as advertised. The first set taught the
assistant's own wording: 456 invented phrases, each stretched into three
sentences. In the second every sentence is written whole and on its own, as a
variation on a way real reviewers worded the report (the dev part of the
labelled rounds, and round 9) or, for the look-alikes, on a sentence a
trained model wrongly scored high. They are kept as plain text, one sentence
a line, under research/synthetic/v2_source/, and none is a copy of a real
sentence (checked when the file is built).

    python research/synthetic_data.py         # writes the first set
    python research/synthetic_data.py --v2    # writes the second set
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "synthetic"
FILE = OUT / "synthetic_sentences_v1.csv"
MANIFEST = OUT / "manifest.json"
SOURCE_V2 = OUT / "v2_source"
FILE_V2 = OUT / "synthetic_sentences_v2.csv"
MANIFEST_V2 = OUT / "manifest_v2.json"
FILES = {"v1": FILE, "v2": FILE_V2}
SEED = 20261009
VARIANTS = 3
LABELS = ("safety", "defect", "quality", "counterfeit", "not_as_described")

PRODUCTS = (
    "charger", "powerbank", "earphones", "earbuds", "speaker", "headset", "cable", "mouse", "keyboard",
    "adapter", "mic", "fan", "smartwatch", "extension", "item", "unit", "product",
)
BRANDS = ("anker", "samsung", "apple", "jbl", "xiaomi", "realme", "oppo", "vivo", "sony", "lenovo", "ugreen", "baseus", "logitech")
BIG = ("20000", "30000", "50000", "10000", "128", "256", "120", "65", "100")
SMALL = ("5000", "3000", "2000", "8", "16", "32", "10", "18", "20")
COMPLAINT_OPEN = (
    "", "", "", "", "grabe", "sayang", "hay nako", "kaso", "pero", "tapos", "seller", "honest review", "update",
    "edit", "ok sana kaso", "disappointed", "nakaka dismaya", "ingat kayo", "warning", "2 stars lang kasi",
)
COMPLAINT_CLOSE = (
    "", "", "", "", "sayang pera", "di na ako uulit", "nakakainis", "pls refund", "wag nyo na bilhin",
    "not recommended", "sana maayos nyo", "po", "talaga", "nakaka disappoint", "ano ba yan", "ibabalik ko to",
)
PLAIN_OPEN = (
    "", "", "", "", "so far", "ok", "update", "honest review", "tbh", "for the price", "sa totoo lang",
    "infairness", "buti", "ayos", "salamat seller", "received",
)
PLAIN_CLOSE = (
    "", "", "", "", "po", "naman", "talaga", "salamat", "thank you seller", "recommended", "sulit", "happy ako",
    "so far so good", "ok na ok", "will order again", "5 stars",
)

# One core per line. "{P}" is a product word, "{B}" a brand, "{N}" a large and
# "{M}" a small number. "## label" at the end of a line adds that label.
CORES: dict[tuple[str, str], str] = {
    ("safety", "positive"): """
        umusok yung {P} habang naka charge
        biglang umusok {P}
        nag spark yung {P} pagkasaksak
        nag-spark sa saksakan
        may spark na lumabas sa port
        sobrang init ng {P} hindi na mahawakan
        uminit ng sobra {P} parang masusunog
        nag overheat {P} kahit 10 mins pa lang
        nag ooverheat palagi
        muntik nang masunog yung {P}
        nasunog yung {P}
        nasunog yung dulo ng cable
        sumabog yung {P}
        pumutok {P} habang naka saksak
        biglang pumutok tapos nangamoy
        lumobo yung battery ng {P}
        namamaga na yung battery
        bloated na battery after 1 month
        nangamoy sunog {P}
        amoy sunog pag naka charge
        may amoy na parang nasusunog na plastic
        nakuryente ako sa {P}
        nakukuryente ako pag hinahawakan habang naka charge
        na ground ako sa {P}
        grounded yung body ng {P}
        may ground pag hinawakan
        nag short circuit {P}
        nag short tapos namatay kuryente sa bahay
        natunaw yung plug ng {P}
        nalusaw yung port sa init
        tunaw yung dulo ng charger
        napaso ako sa init ng {P}
        napaso daliri ko
        nag apoy yung {P}
        nagliyab yung {P}
        may lumabas na usok sa {P}
        umiinit tapos umaamoy
        nag init ng todo yung adapter
        the {P} exploded while charging
        it caught fire
        started smoking after a few minutes
        got so hot it burned my hand
        the port melted
        battery swelled up
        bulging battery already
        gave me an electric shock
        sparks came out when i plugged it in
        there is a burning smell
        it overheats badly
        too hot to touch while charging
        fire hazard itong {P}
        delikado sobrang init ng {P}
        muntik na magka sunog sa bahay dahil sa {P}
        nagka sunog sa extension dahil dito
        umuusok na yung wire
        nangingitim yung saksakan sa init
        may tumalsik na spark
        kumislap tapos umusok
        almost burned my table
        smoke came out of the adapter
    """,
    ("safety", "negative"): """
        hindi naman umiinit kahit matagal gamitin
        di nag iinit {P} kahit naka fast charge
        walang amoy sunog safe gamitin
        hindi nag overheat
        malamig pa rin {P} kahit ilang oras
        wala namang spark pag sinasaksak
        hindi naman nakaka kuryente
        safe naman gamitin {P}
        ang init ng panahon kaya sakto tong fan
        mainit sa kwarto kaya bumili ako ng fan
        hot item to sa tiktok kaya binili ko
        sunog ang bulsa ko pero worth it
        nasira na yung luma kong charger kaya bumili ako nito
        umuusok sa bilis ang delivery
        apoy ang tunog ng {P} solid
        lit ang bass grabe
        nakaka shock ang ganda ng quality
        shock ako ang bilis dumating
        na shock ako sa laki ng box
        mabilis uminit yung heater ang ganda
        mabilis uminit ang plantsa kaya ok
        pang init ng tubig ok naman
        smoke effect ng humidifier ang ganda
        mainit pa yung parcel galing sa araw
        baka uminit pag matagal pero di ko pa nasusubukan
        akala ko sasabog sa mura pero ok naman pala
        takot ako baka mag overheat pero so far hindi naman
        no overheating so far
        does not get hot even after hours
        no burning smell at all
        it stays cool while charging
        the sound is fire
        this deal is hot
        blown away by the bass
        explosive bass ang lakas
        pampasabog ng tenga sa lakas ng volume
        color fire red ang dumating
        flame design ng case ang ganda
        hindi naman lumobo battery after 3 months
        walang ground pag hinawakan
        di naman ako nakuryente
        hindi umuusok hindi umiinit perfect
        tested for 5 hours walang init
        mabilis mag charge pero hindi umiinit
        wala namang amoy
        sana hindi uminit pag matagal
        umiinit ba to pag naka charge
    """,
    ("counterfeit", "positive"): """
        fake yung {P} hindi original
        peke to wag kayo maniwala
        class a lang to hindi orig
        replica lang yung {P}
        imitation lang pala
        hindi authentic yung {P}
        hindi tunay na {B}
        fake na {B} yung dumating
        copy lang ng {B}
        kopya lang hindi legit
        oem lang hindi original gaya ng sabi
        hindi legit na {B} to
        pekeng {B}
        fake ang logo iba sa orig
        halatang fake yung tatak
        iba ang serial hindi nag verify sa website
        di ma verify sa official site peke
        walang serial number fake
        counterfeit item
        this is a fake {B}
        not genuine
        obviously a knockoff
        cheap imitation of the original
        its a replica not the real one
        clone lang ng {B}
        super copy lang
        hindi galing sa {B} mismo fake
        sabi original pero class a pala
        nag compare ako sa orig ko ibang iba fake to
        malayo sa original ko peke
        hindi kagaya ng original na {B}
        fake packaging at fake item
        tinest ko sa app fake daw
        na check sa store peke daw
        sabi ng technician fake daw
        pinacheck ko sa mall hindi daw orig
        walang hologram peke
        mali spelling ng brand halatang peke
        {B} daw pero fake
        legit daw pero peke naman
        fake na {B} scam
        not an original {B} product
        definitely not authentic
        fake na earbuds hindi orig
        hindi original na charger ng {B}
        di orig yung battery
        imitation ng {B} lang
        bootleg lang to
        hindi totoong {B}
        mukhang orig pero peke pala
    """,
    ("counterfeit", "negative"): """
        legit na {B} to
        original talaga yung {P}
        authentic na authentic
        hindi fake legit seller
        akala ko fake pero original pala
        na verify ko sa website legit
        may serial number at na verify
        orig to promise
        100% original
        genuine product
        tunay na {B}
        not fake at all
        legit seller mabilis mag ship
        mukhang original naman
        original ang packaging
        sabi ng iba fake pero sakin legit
        duda ako nung una pero legit pala
        baka fake kaya di ko pa ginagamit
        sana original to
        paano malalaman kung fake
        original ba to seller
        fake news yung sinabi ng iba
        walang fake dito
        hindi ito class a original talaga
        generic lang pero ok naman
        unbranded pero gumagana
        scam seller hindi nag deliver
        scam walang dumating
        budol ang seller kulang ang order
        hindi totoo yung {N}mah ## not_as_described
        iba sa picture yung dumating ## not_as_described
        legit ang bilis ng delivery
        legit ang ganda
        original price mahal pero sale ngayon
        orig na box may seal
        the real deal
        authentic item fast delivery
        no doubt original
        passed the authenticity check
        kahit mura legit naman
        parang original ang quality
        hindi peke promise
    """,
    ("not_as_described", "positive"): """
        hindi totoo yung {N}mah
        {N}mah daw pero {M}mah lang talaga
        sabi {N}w pero {M}w lang
        hindi naman pala fast charge gaya ng sabi
        wala palang bluetooth kahit nakalagay
        walang noise cancelling gaya ng sabi
        hindi waterproof kahit nakalagay
        iba sa picture yung dumating
        malayo sa picture
        hindi kamukha ng nasa picture
        iba itsura sa actual
        mas maliit kesa sa picture
        ang liit pala hindi gaya sa description
        hindi tugma sa description
        mali ang description
        misleading yung description
        misleading ang ads
        false advertising
        not as described
        not as advertised
        different from the photos
        smaller than what the listing says
        capacity is not what they claim
        only {M}gb not {N}gb
        {N}gb daw pero {M}gb lang ang laman
        fake capacity
        fake ang mah
        hindi totoo ang battery capacity
        sabi 2 meters pero 1 meter lang
        maikli hindi 2m gaya ng nakasulat
        walang rgb kahit nasa picture
        walang mic kahit sabi may mic
        sabi may fast charging wala naman
        hindi {N}w normal charge lang
        hindi totoo ang {N}w
        sabi wireless pero may wire
        hindi touch control gaya ng sabi
        hindi pala compatible sa iphone kahit nakalagay
        sabi pwede sa laptop hindi naman
        hindi umaabot ng 10 hours gaya ng sabi
        2 hours lang hindi 8 hours gaya ng nakalagay
        specs sa listing hindi totoo
        iba ang specs sa actual
        hindi totoo ang nakasulat sa box
        exaggerated ang description
        yung nasa video iba sa dumating
        sa picture malaki sa actual maliit
        not the same as shown
        listing says {N}w but it is {M}w
        hindi sya bluetooth 5.3 gaya ng sabi
        kulang ang capacity sa sinabi
        kalahati lang ng sinabing capacity
    """,
    ("not_as_described", "negative"): """
        same as picture
        gaya ng nasa picture
        katulad ng description
        legit ang {N}mah
        totoo ang capacity
        as described
        exactly as advertised
        true to the photos
        tugma sa description
        kung ano nasa picture yun ang dumating
        mas maganda pa sa picture
        mas malaki pa kesa sa inaakala ko
        {N}mah talaga tested
        accurate ang specs
        walang daya sa description
        hindi misleading
        honest ang seller sa description
        sakto sa sukat na nakalagay
        {M}mah lang battery ng phone ko kaya ilang charge
        {M}w lang charger ko dati kaya upgrade to
        maliit lang pero yun naman talaga nasa description
        alam ko namang maliit kasi nakalagay
        di ko nabasa description kaya akala ko malaki
        akala ko wireless yun pala wired kasalanan ko di ko binasa
        mali ang napindot kong variant
        mali ang item na pinadala
        kulang ang dumating isa lang
        iba ang kulay na pinadala hindi yung inorder ko
        wala yung freebie
        baka hindi totoo ang capacity pero di ko pa na test
        sana totoo ang {N}mah
        totoo ba ang capacity nito
        the photos are accurate
        matches the listing
        specs are legit
    """,
    ("quality", "positive"): """
        mahina ang tunog ng {P}
        ang hina ng bass
        mahina ang volume kahit max
        mahina ang hangin ng fan
        mahina ang ilaw
        mahina sumagap ng signal
        malabo ang tunog
        sabog ang tunog pag malakas
        basag ang sound
        matinis walang bass
        garalgal ang mic
        manipis ang wire
        marupok ang plastic
        flimsy ang pagkakagawa
        parang laruan lang
        cheap ang materials
        mumurahin ang dating
        low quality ang plastic
        poor quality
        hindi matibay tingnan
        magaan masyado parang hungkag
        maluwag ang takip
        may gasgas pagdating
        may dent yung {P}
        may yupi sa gilid ng {P}
        may scratches ang screen
        may lamat yung case
        may mantsa yung {P}
        kupas ang kulay
        tabingi ang print
        hindi pantay ang pagkakadikit
        magaspang ang finish
        matigas pindutin ang buttons
        mabagal mag connect pero gumagana
        medyo delay ang sound pero ok lang
        gumagana naman pero mahina
        ok lang pero hindi ganun kaganda ang quality
        sound quality is poor
        feels cheap
        very flimsy build
        weak bass
        the mic sounds muffled
        arrived with scratches
        has a small dent but works
        build quality is bad
        plastic feels brittle
        volume is too low
        thin cable
        the light is dim
        pangit ang pagkakagawa
        mahinang klase ang materyales
        parang madaling mabali ang hinge
    """,
    ("quality", "negative"): """
        malakas ang tunog
        ang lakas ng bass
        malinaw ang sound
        matibay ang {P}
        makapal ang wire
        premium ang feel
        solid ang build
        walang gasgas pagdating
        walang dent maayos ang packaging
        hindi naman mahina ang tunog
        hindi mumurahin tingnan
        maganda ang quality
        quality is great
        sturdy build
        loud and clear
        malakas ang hangin
        maliwanag ang ilaw
        yupi ang box pero ok ang item
        sira ang kahon pero buo ang laman
        gasgas lang ang karton
        mahina ang signal sa area namin kaya di ko ma test
        mahina ang wifi namin
        mahina ang kuryente dito sa probinsya
        mahina ako sa tech kaya nahirapan mag setup
        mura pero hindi cheap tingnan
        sabi ng iba mahina pero sakin malakas naman
        baka marupok pero so far ok
        sana matibay
        matibay kaya to
        no scratches
        not flimsy at all
        feels premium
        strong bass
        clear mic
    """,
    ("defect", "positive"): """
        ok naman pero minsan nag hahang
        maganda sana kaso nag didisconnect minsan
        ok lang pero mabilis ma lowbat
        gumagana pero yung isang side ayaw na
        nagloloko minsan ang bluetooth
        medyo mabagal mag charge
        minsan ayaw mag on
        namamatay bigla minsan
        kusang nag ooff
        nawawala ang sound sa kaliwa
        one side stopped working
        sometimes it disconnects
        charges slowly
        battery drains fast
        the left earbud is dead
        stops working after a while
        ayaw na mag charge after 1 week
        hindi na nag oon
        nasira agad after 3 days
        di gumagana ang mic
        putol putol ang tunog
        nag stop na gumana
        sira yung isang port
        hindi na ma charge
        dead on arrival
    """,
    ("defect", "negative"): """
        gumagana naman sya
        gumagana ng maayos
        working naman lahat
        nag chacharge naman ng maayos
        fast charging naman pala
        full charge agad
        na full charge ko na
        tinry ko i charge ng 2 hours
        chinarge ko muna bago gamitin
        charging speed 100
        charging speed ok
        battery life di ko pa na test
        di ko pa nagagamit
        hindi ko pa na try
        pano ba to i charge
        paano i connect sa phone
        paano gamitin to
        saan pinipindot para mag on
        do not buy from this store
        bad seller
        hindi nag rereply ang seller
        late ang delivery
        matagal dumating
        mali ang item na pinadala
        kulang ang order
        ibang kulay ang dumating
        mabilis ma charge matagal ma lowbat
        matagal malowbat ang ganda
        hindi madaling ma lowbat
        di naman nag hahang
        walang sira pagdating
        wala namang problema
        no issues so far
        works perfectly
        still working after 3 months
        hindi pa nasisira
        hindi naman nasira kahit nabagsak
        akala ko sira lowbat lang pala
        akala ko di gumagana di ko lang pala na on
        pag nasira bibili ulit ako
        sana tumagal
        sana hindi masira agad
        baka masira pag nabasa
        kung masira may warranty naman
        lowbat pagdating pero normal lang
        naka off lang pala
        na drain ko muna bago i charge
        it charges my phone twice
        charged it overnight
        i charged it first
        does it charge laptops
        how do i turn it on
        turns on and off easily
        the off button is on the side
        not broken
        nothing wrong with it
        battery lasts long
        does not lag
        no disconnects
    """,
}


def cores() -> list[dict[str, Any]]:
    rows = []
    for (family, kind), block in CORES.items():
        for line in block.splitlines():
            text, _, extra = line.strip().partition("##")
            if not text.strip():
                continue
            labels = {label: False for label in LABELS}
            if kind == "positive":
                labels[family] = True
                # The owner's rule of 7 October: a safety report is a defect report as well.
                labels["defect"] = labels["defect"] or family == "safety"
            if extra.strip():
                labels[extra.strip()] = True
            rows.append({"family": family, "kind": kind, "core": text.strip(), "labels": labels})
    return rows


def sentences() -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    out: list[dict[str, Any]] = []
    for number, core in enumerate(cores(), start=1):
        seen: set[str] = set()
        attempts = 0
        while len(seen) < VARIANTS and attempts < 30:
            attempts += 1
            body = (
                core["core"].replace("{P}", rng.choice(PRODUCTS)).replace("{B}", rng.choice(BRANDS))
                .replace("{N}", rng.choice(BIG)).replace("{M}", rng.choice(SMALL))
            )
            # A report is usually framed as a complaint and a look-alike plainly,
            # but one time in six the frames swap, so a frame never marks a label.
            complaint = (core["kind"] == "positive") != (rng.random() < 1 / 6)
            opening = rng.choice(COMPLAINT_OPEN if complaint else PLAIN_OPEN)
            closing = rng.choice(COMPLAINT_CLOSE if complaint else PLAIN_CLOSE)
            text = " ".join(part for part in (opening, body, closing) if part)
            style = rng.random()
            text = text.capitalize() if style < 0.45 else text.upper() if style > 0.95 else text
            text += rng.choice(("", "", ".", ".", "!", "..", "!!"))
            if text.casefold() in seen:
                continue
            seen.add(text.casefold())
            out.append({
                "row_id": f"syn-{len(out) + 1:05d}", "core_id": f"core-{number:04d}", "family": core["family"],
                "kind": core["kind"], "text": text, **{label: core["labels"][label] for label in LABELS},
            })
    return out


def written_v2() -> list[dict[str, Any]]:
    """The second set: every line of research/synthetic/v2_source/ as a row."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(SOURCE_V2.glob("*.txt")):
        family, _, kind = path.stem.rpartition("_")
        for line in path.read_text(encoding="utf-8").splitlines():
            text, _, extra = line.strip().partition("##")
            text = text.strip()
            if not text or text.startswith("#") or text.casefold() in seen:
                continue
            seen.add(text.casefold())
            labels = {label: False for label in LABELS}
            if kind == "positive":
                labels[family] = True
            if extra.strip():
                labels[extra.strip()] = True
            # The owner's rule of 7 October: a safety report is a defect report as well.
            labels["defect"] = labels["defect"] or labels["safety"]
            out.append({
                "row_id": f"syn2-{len(out) + 1:05d}", "core_id": f"v2-{len(out) + 1:05d}", "family": family,
                "kind": kind, "text": text, **labels,
            })
    return out


def load_synthetic_records(sets: tuple[str, ...] = ("v1",)) -> list[dict[str, Any]]:
    """The frozen files as training records, shaped like five_label_data's.

    The first set alone by default, so the scripts written for it give the
    figures they gave; pass ("v1", "v2") for both.
    """
    rows: list[dict[str, str]] = []
    for name in sets:
        with FILES[name].open(encoding="utf-8", newline="") as source:
            rows += list(csv.DictReader(source))
    return [
        {
            "row_id": row["row_id"], "review_id": row["core_id"], "text": row["text"], "round": "synthetic",
            "team": {}, "labels": {label: row[label] == "yes" for label in LABELS},
            "gadget": True, "part": "synthetic", "stratum": f"{row['family']}_{row['kind']}", "rating": "", "weight": 1.0,
        }
        for row in rows
    ]


def main_v2() -> int:
    """Write the second set, leaving out any line that repeats a real sentence."""
    from five_label_data import load_five_label_records

    def plain(text: str) -> str:
        return " ".join("".join(ch if ch.isalnum() else " " for ch in text.casefold()).split())

    real = {plain(record["text"]) for record in load_five_label_records(include_live_check=True)}
    written = written_v2()
    rows = [row for row in written if plain(row["text"]) not in real]
    for number, row in enumerate(rows, start=1):
        row["row_id"], row["core_id"] = f"syn2-{number:05d}", f"v2-{number:05d}"
    with FILE_V2.open("w", encoding="utf-8", newline="\n") as target:
        writer = csv.DictWriter(target, fieldnames=["row_id", "core_id", "family", "kind", "text", *LABELS], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **{label: "yes" if row[label] else "no" for label in LABELS}})
    families = sorted({(row["family"], row["kind"]) for row in rows})
    manifest = {
        "what": "Second set of training sentences written by the AI assistant, one by one. Not reviews. Training only, never a test.",
        "approved_by": "adviser, as reported by the owner on 8 October 2026",
        "source": "research/synthetic/v2_source/*.txt", "sentences": len(rows),
        "left_out_because_identical_to_a_real_sentence": len(written) - len(rows),
        "by_family": {f"{family}_{kind}": sum(1 for row in rows if (row["family"], row["kind"]) == (family, kind)) for family, kind in families},
        "yes_per_label": {label: sum(1 for row in rows if row[label]) for label in LABELS},
        "sha256": hashlib.sha256(FILE_V2.read_bytes()).hexdigest(),
    }
    MANIFEST_V2.write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(manifest, indent=2))
    return 0


def main() -> int:
    if "--v2" in sys.argv:
        return main_v2()
    rows = sentences()
    OUT.mkdir(parents=True, exist_ok=True)
    with FILE.open("w", encoding="utf-8", newline="\n") as target:
        writer = csv.DictWriter(target, fieldnames=["row_id", "core_id", "family", "kind", "text", *LABELS], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **{label: "yes" if row[label] else "no" for label in LABELS}})
    counts = {
        f"{family}_{kind}": sum(1 for row in rows if row["family"] == family and row["kind"] == kind)
        for family, kind in CORES
    }
    manifest = {
        "what": "Training sentences written by the AI assistant. Not reviews. Training only, never a test.",
        "approved_by": "adviser, as reported by the owner on 8 October 2026",
        "seed": SEED, "variants_per_core": VARIANTS, "cores": len(cores()), "sentences": len(rows),
        "by_family": counts, "yes_per_label": {label: sum(1 for row in rows if row[label]) for label in LABELS},
        "sha256": hashlib.sha256(FILE.read_bytes()).hexdigest(),
    }
    MANIFEST.write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
