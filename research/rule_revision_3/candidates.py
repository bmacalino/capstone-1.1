"""Third rule revision: candidate patterns, each checked on sentences it was
not written from.

  group "r2"   the not-as-advertised patterns dropped on 8 Oct, written from
               the dev part of rounds 1-8  -> checked on round 9
  group "r9"   patterns written from the round 9 misses -> checked on the dev
               part of rounds 1-8 (and the live-check listings)

Rule, fixed before running: a pattern is kept only if on its check sentences
it adds at least one right flag and at least 3 in 4 of the flags it adds are
right. A pattern that matches nothing there is dropped: no evidence.
The held-out third is not touched here.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = str(Path(__file__).resolve().parents[2])
sys.path[:0] = [ROOT, ROOT + r"\sentiment-analysis", ROOT + r"\research"]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from defaketive_sentiment import extra_labels as current  # noqa: E402
from defaketive_sentiment import text_norm  # noqa: E402
from five_label_data import load_five_label_records  # noqa: E402
from improve_label_rules import held_out  # noqa: E402

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("extra_labels_r2full", HERE / "extra_labels_r2full.py")
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)

N, U, F, NOT, SHOWN, SOFT = full._NUM, full._UNIT, full._FILL, full._NOT, full._SHOWN, full._SOFT
LISTING = r"(?:pictures?|pics?|photos?|litrato|description|discription|thumbnail|ads?|advertisement|post|caption|product info\w*|listing)"

# ---- not as advertised, written from round 9 --------------------------------
NAD_R9 = {
    "num_nakalagay": rf"{N}\s*(?:{U}\s+)?(?:ang\s+|yung\s+|ung\s+)?(?:nakalagay|naka lagay|naka lagat|nkalagay|nakasulat)(?!\w)",
    "nakalagay_num": rf"(?:nakalagay|naka lagay|nkalagay|nakasulat)\s+{F}{{0,2}}{N}(?!\w)",
    "sabi_sa_listing": rf"(?:sabi|sabe)\s+(?:po\s+)?(?:nila\s+|nyo\s+|niyo\s+)?sa\s+{LISTING}(?!\w)",
    "nakalagay_sa_listing": rf"(?:nakalagay|naka lagay|nkalagay|nakasulat|naka sulat|sinasabi|indicated|stated)\s+{F}{{0,2}}(?:sa|on|in)\s+(?:the\s+)?(?:description|discription|ads?|advertisement|post|caption|specs?|product info\w*|manual|listing)(?!\w)",
    "hindi_tugma": rf"{NOT}\s+{SOFT}{{0,2}}(?:tugma|match|kamukha|akma)\s+{F}{{0,2}}{SHOWN}(?!\w)",
    "shown_hindi_kamukha": rf"{SHOWN}\s+{F}{{0,3}}{NOT}\s+(?:kamukha|katulad|kapareho|tugma)(?!\w)",
    "iba_nasa_picture": rf"(?:ibang iba|iba|ibang)\s+{F}{{0,4}}(?:nasa|asa|nsa|na sa)\s+{SHOWN}(?!\w)",
    "hindi_nasa_picture": rf"{NOT}\s+{F}{{0,3}}(?:nasa|asa|nsa)\s+{SHOWN}(?!\w)",
    "sa_picture_may": rf"sa\s+{SHOWN}\s+(?:may|meron|merong)(?!\w)",
    "same_as_advertise": r"(?:same|different)\s+(?:as|from|sa)\s+(?:the\s+)?advertise\w*",
    "as_advertised_compare": rf"(?:compared to|instead of|supposed to be)\s+{F}{{0,3}}as advertised(?!\w)",
    "different_from_listed": rf"(?:different|off|far)\s+from\s+(?:what|the)\s+{F}{{0,3}}(?:listed|shown|showing|posted|advertised|pictured)(?!\w)",
    "ordered_num_received": rf"ordered\s+{F}{{0,2}}{N}\s*{U}\s+{F}{{0,6}}(?:received|sent|got|dumating|binigay)\s+{F}{{0,3}}{N}",
    "only_received_num": rf"only\s+(?:received|got|have|has)\s+{F}{{0,2}}{N}\s*{U}(?!\w)",
    "num_not_num": rf"{N}\s*(?:\"|''|'|{U})?\s*not\s+{N}(?!\w)",
    "hindi_totoo_wide": rf"{NOT}\s+{SOFT}{{0,3}}(?:kase\s+|kasi\s+|kc\s+)?(?:totoo|totoong|totong|toto?ong)(?!\w)",
    "hindi_talagang_num": rf"{NOT}\s+{SOFT}{{0,2}}(?:talagang|tlgang|mukhang|mukang|totong)\s+{N}\s*{U}(?!\w)",
    "akala_hindi_pala": rf"(?:akala|kala)\s*(?:ko|q|ku\w*)?\s+{F}{{0,8}}(?:{NOT}|wala)\s+(?:naman\s+|nman\s+)?(?:pala|pla)(?!\w)",
    "akala_feature": r"(?:akala|kala)\s*(?:ko|q|ku\w*)?\s+(?:[\w']+\s+){0,2}(?:rec+hargea?ble|wireless|bluetooth|sensor|waterproof|touch ?screen)(?!\w)",
    "i_thought_feature": r"i thought\s+(?:[\w']+\s+){0,6}(?:compatible|bluetooth|wireless|rechargeable|included)(?!\w)",
    "fast_charger_typo": rf"{NOT}\s+{SOFT}{{0,3}}fast ?(?:chrg|carg|charh)\w*",
    "not_compatible": rf"{NOT}\s+{SOFT}{{0,2}}compatible(?!\w)",
    "parang_normal_lang": r"(?:parang|para|just like an?)\s+(?:[\w']+\s+){0,2}(?:normal|ordinary|regular|mini|maliit)\s+(?:na\s+)?(?:electric\s+)?(?:charger|fan|mic|speaker|mouse)(?!\w)",
    "versus_reality": r"(?:description|pictures?|expectations?)\s+(?:vs\.?|versus)\s+reality(?!\w)",
    "malau_sa_picture": rf"(?:malau|malayu|malayong malayo)\s+{F}{{0,3}}sa\s+{F}{{0,2}}{SHOWN}(?!\w)",
    "box_says": rf"(?:box|description|listing|label|ad|manual|packaging)\s+says?\s+{F}{{0,3}}(?:{N}|cat|type|fast|with)",
    "ang_totoo_num": rf"ang totoo\s+{F}{{0,2}}{N}\s*{U}(?!\w)",
    "fake_comma_num": rf"(?:fake|peke),\s+{F}{{0,2}}{N}\s*{U}(?!\w)",
    "lagay_sa_description": r"(?:paglagay|nilagay|ilagay|put it|pakiedit|not included|di kasi included)\s+(?:[\w']+\s+){0,4}(?:description|discription|caption)(?!\w)",
    "hindi_ganto_picture": rf"{NOT}\s+{F}{{0,2}}(?:ganto|ganyn|gnito|ganitong)\s+{F}{{0,6}}{SHOWN}(?!\w)",
    "kasi_sa_picture": rf"(?:kse|kasi|kc|pla|pala)\s+{F}{{0,2}}sa\s+{SHOWN}\s+(?:medyo|mejo|parang|mukhang)(?!\w)",
}

# ---- poor quality, written from round 9 -------------------------------------
_Q_BAD = (
    r"(?:not\s+(?:(?:really|very|so|that|too|as|the)\s+)?(?:good|great|best|loud|clear)|awful|bad|poor|worst|terrible|pangit|panget"
    r"|low|basag|sabog|mahina|kulang|nakaka ?disappoint\w*)"
)
QUALITY_R9 = {
    "quality_is_bad_wide": rf"qualit\w*\s*:?\s*(?:(?:is|was|are|nya|niya|sya|siya)\s+|of\s+(?:the\s+)?\w+\s+is\s+|ng\s+\w+\s+)?(?:(?:very|bery|so|really|too)\s+)?{_Q_BAD}(?!\w)",
    "material_bad": r"materials?\s*:?\s*(?:is |feels? |are |nya |niya )?(?:poor|cheap|mura lang|pangit|panget|manipis)(?!\w)",
    "lo_quality": r"(?:lo|low|poor|bad)\s+(?:sound\s+)?(?:qualit\w*|sound system)(?!\w)",
    "hindi_sya_quality": rf"{NOT}\s+(?:sya|siya|xa)\s+quality(?!\w)",
    "disappointed_quality": r"disappointed\s+(?:with|sa|in)\s+(?:the\s+)?quality(?!\w)",
    "bass_lacking": r"(?:kulang\s+(?:ng|sa|ang|yung)\s+bass|walang\s+bass|lacks?\s+bass|no\s+bass|terrible\s+bass|hindi\s+(?:[\w']+\s+){0,3}may\s+bass)(?!\w)",
    "echo_kulob": r"(?:ma ?echo|may echo|echoes|kulob|ngongo|ngungu|garargar|garal gal)(?!\w)",
    "not_clear": rf"{NOT}\s+(?:(?:sya|siya|naman|gaanong|ganun|ganon|masyadong|very|too|so|that)\s+){{0,2}}(?:clear|cleae|malinaw|kalinaw|maaninag)(?!\w)",
    "malabo_wide": r"(?:malabo\s+(?:ang\s+|yung\s+|ung\s+)?(?:music|cam|picture|display)|(?:medyo|mejo)\s+malabo)(?!\w)",
    "hindi_kalakas_wide": rf"{NOT}\s+(?:(?:pala|rin|din|na|nga|lang|lng|sya|siya|ganong|ganon|ganun|gaano|naman|nman|masyado|masyadong|gaanong|too|very|that|so)\s+){{1,4}}(?:ka)?(?:lakas|malakas|kalakasan|loud)(?!\w)",
    "hindi_malamig": rf"{NOT}\s+(?:(?:naman|nman|sya|siya|masyadong|ganun|gaanong|rin|din)\s+){{0,2}}(?:ka)?(?:lamig|malamig)(?!\w)",
    "malakas_pa": r"(?:(?:mas\s+)?malakas pa\s+(?:[\w']+\s+){0,6}(?:kesa|kaysa)|mas malakas pa\s+(?:pala\s+|psla\s+)?(?:ang|yung|ung))(?!\w)",
    "weak_words": r"(?:may kahinaan|very weak|weak (?:sound|magnet\w*|signal|bass)|(?:sound|volume|bass)\s+is\s+(?:very |so |too )?weak|cheap sound|parang bulong)(?!\w)",
    "mabagal": r"(?:mabagal|bagal)\s+(?:lang\s+)?(?:ang|yung|ung|sya|siya|response)(?!\s+(?:[\w']+\s+){0,2}(?:deliver\w*|shipp?\w*|dating|pagdating|dumating|courier|rider|charg\w*|mag ?charge|magcharge|seller|reply|mapuno|ma ?full))(?!\w)",
    "maluwag_wide": rf"(?:(?:maluwag|maluwang|loose)\s+{F}{{0,3}}(?:socket|lagayan|saksakan|port|takip|cover)|(?:socket|saksakan|port|lagayan|takip)\s+{F}{{0,3}}(?:maluwag|maluwang|loose))(?!\w)",
    "falls_over": r"(?:natatanggal|nabubuwal|natutumba|matutumba|umuuga|umaalog)(?!\w)",
    "tumutunog_sa_loob": r"tumutunog\s+(?:[\w']+\s+){0,3}loob(?!\w)",
    "hindi_maayos_gawa": rf"{NOT}\s+(?:(?:rin|din|po|lang|masyadong)\s+){{0,2}}maayos\s+(?:(?:ang|yung|ung)\s+)?(?:pagkakabuo|pagkakagawa|gawa|saksakan|pagka ?welding)(?!\w)",
    "fit_words": rf"{NOT}\s+(?:(?:sya|siya|naman|gaanong|rin|din|lang)\s+){{0,2}}(?:pangmatagalan|pang matagalan|lumalapat|patag|eksakto|nakasya|magkaparehas)(?!\w)",
    "may_basag_part": r"(?:may|my|meron|merong)\s+basag\s+(?:sa\s+)?(?:gilid|kanto|ilalim|likod|yung isa|ung isa)(?!\w)",
    "cosmetic_more": r"(?:scatches|may uka|na- ?bend|may gas2|lubog)(?!\w)",
    "sobrang_gaan": r"(?:sobrang|subrang|masyadong)\s+gaan(?!\w)",
    "hirap_wide": r"(?:hirap\s+(?:pa\s+|nyang\s+|niyang\s+|rin\s+|din\s+)?(?:ilagay|idrive|i ?bluetoo?th|mag ?insert|isuot|gamitin)|mahirap\s+(?:rin\s+|din\s+)?mag ?insert|stiff|magaspang)(?!\w)",
    "lag_words": r"(?:may delay(?!\s+(?:sa|ang|yung|ung)\s+(?:deliver\w*|shipp?\w*|dating|pagdating))|naglalag|nag ?lalag|doble ang tunog|white noise)(?!\w)",
    "pangit_wide": rf"(?:pangit|panget)\s+{F}{{0,2}}(?:kabuuan|tignan|tingnan|chord|cord|cable|wire)(?!\w)",
    "bilis_sira_wide": r"(?:madali|madaling|mabilis|dali|bilis)\s+(?:(?:lang|lng|na|ma|sya|siya|nya|nga|po)\s+){0,3}(?:sira|masira|makalas|matumba|mabali|bumigay)(?!\w)",
}

records = load_five_label_records(include_live_check=True)
norm = [text_norm.normalize_with_spans(r["text"])[0] for r in records]
rnd = np.array([str(r["round"]) for r in records])
held = np.array([held_out(r) for r in records])
ROUND9 = rnd == "9"
DEV = ~held & ~ROUND9


def truth_of(label):
    return np.array([bool(r["labels"].get(label)) for r in records])


def nad_matches(pattern):
    compiled = re.compile(rf"(?<!\w)(?:{pattern})", re.IGNORECASE)
    out = []
    for text in norm:
        hit = False
        for found in compiled.finditer(text):
            before = text[: found.start()]
            if full.SPECULATION.search(before) or "dual shock" in found.group(0).casefold():
                continue
            if full.ATTRIBUTIVE_BEFORE.search(before) and full.STARTS_NEGATED.match(found.group(0)):
                continue
            hit = True
            break
        out.append(hit)
    return np.array(out)


def quality_matches(pattern):
    compiled = re.compile(rf"(?<!\w)(?:{pattern})", re.IGNORECASE)
    out = []
    for text in norm:
        hit = False
        for found in compiled.finditer(text):
            before = text[: found.start()]
            if (current.NEGATED_BEFORE.search(before) or current.NEGATED_INSIDE.search(found.group(0))
                    or current.HEARSAY_BEFORE.search(before) or current.CONDITION_BEFORE.search(before)):
                continue
            hit = True
            break
        out.append(hit)
    return np.array(out)


def report(label, name, hits, base, truth, source, check, show):
    add = hits & ~base
    s = (int((add & truth & source).sum()), int((add & ~truth & source).sum()))
    c = (int((add & truth & check).sum()), int((add & ~truth & check).sum()))
    keep = c[0] >= 1 and c[0] >= 3 * c[1]
    print(f"{'KEEP' if keep else 'drop'}  {label[:7]:7} {name:26} written from: +{s[0]} right, {s[1]} wrong | check: +{c[0]} right, {c[1]} wrong")
    if show:
        for i in np.flatnonzero(add & ~truth & source):
            print(f"        wrong in source: {norm[i][:170]}")
    return {"label": label, "name": name, "source": s, "check": c, "keep": bool(keep)}


def main():
    show = "--show" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    results = []
    base = np.array([bool(current.not_as_described_spans(t)) for t in norm])
    truth = truth_of("not_as_described")
    if only in (None, "r2"):
        # The dropped revision: the patterns listed after its marker comment,
        # and as one more "pattern" the changes it made to the shared pieces.
        old = np.array([bool(full.not_as_described_spans(t)) for t in norm])
        first = len(current.NOT_AS_DESCRIBED_PATTERNS)
        for number, compiled in enumerate(full.NOT_AS_DESCRIBED_PATTERNS[first:], start=1):
            pattern = compiled.pattern[len(r"(?<!\w)(?:"):-1]
            results.append(report("not_as_described", f"r2_{number:02d}", nad_matches(pattern), base, truth, DEV, ROUND9, False))
        shared = np.zeros(len(norm), dtype=bool)
        for compiled in full.NOT_AS_DESCRIBED_PATTERNS[:first]:
            shared |= nad_matches(compiled.pattern[len(r"(?<!\w)(?:"):-1])
        results.append(report("not_as_described", "r2_shared_pieces", shared, base, truth, DEV, ROUND9, False))
        results.append(report("not_as_described", "r2_all_together", old, base, truth, DEV, ROUND9, False))
    if only in (None, "nad"):
        for name, pattern in NAD_R9.items():
            results.append(report("not_as_described", name, nad_matches(pattern), base, truth, ROUND9, DEV, show))
    if only in (None, "quality"):
        base = np.array([bool(current.quality_spans(t)) for t in norm])
        truth = truth_of("quality")
        for name, pattern in QUALITY_R9.items():
            results.append(report("quality", name, quality_matches(pattern), base, truth, ROUND9, DEV, show))
    (HERE / "r3_candidates.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
