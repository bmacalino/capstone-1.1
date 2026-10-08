"""Rules for the two labels added in tagging guide 1.4.

  quality           the product works, but poorly; feels cheap or flimsy; or
                    arrived dented, scratched or torn
  not_as_described  a size, capacity, feature or look differs from what the
                    listing says or shows

They sit beside the three badge categories (safety, defect, counterfeit) and
never change them. Everything here works on normalized text
(`text_norm.normalize_with_spans`), where "nasira" is already "sira" and
"charging" is "charge".

First written from labelling rounds 5-6 and revised on rounds 1-6
(research/RESULTS_TABLES.md, Tables 11i-11j). Round 7 was kept out of the
revision so there is one round the patterns were not fitted to.
"""

from __future__ import annotations

import re

QUALITY = "quality"
NOT_AS_DESCRIBED = "not_as_described"
EXTRA_TAGS = (QUALITY, NOT_AS_DESCRIBED)
# Part of the checker's version string, so saved reports are redone when the
# patterns change. r2 = the second revision of 8 Oct 2026, which changed the
# poor-quality patterns. A revision of the not-as-advertised patterns was
# written at the same time and dropped: on held-out sentences it found nothing
# new (research/label_rules_revision.json).
EXTRA_RULES_REVISION = "r2"

_NOT = r"(?:hindi|hinde|di|d|nd|ndi|not|isn't|isnt|wasn't|wasnt|ain't|aint|doesn't|doesnt|never)"
# The poor-quality patterns also read three common misspellings of "hindi".
_NOT_WIDE = r"(?:hindi|hinde|hind|hndi|hnd|di|d|nd|ndi|not|isn't|isnt|wasn't|wasnt|ain't|aint|doesn't|doesnt|never)"
_FILL = r"(?:[\w']+\s+)"
_NUM = r"\d[\d,.]*"
_UNIT = r"(?:k\s*)?(?:mah|gb|tb|mbps|watts?|w|meters?|mtrs?|m|cm|liters?|litter|l|ml|inch(?:es)?|ribs)"
_SHOWN = r"(?:pictures?|pics?|photos?|litrato|display|description|discription|thumbnail|ads?|post)"
_SOFT = r"(?:(?:po|naman|nman|talaga|tlga|sya|siya|cya|xa|sia|rin|din|a|even|really|totally)\s+)"
_DEGREE = r"(?:(?:sya|siya|xa|cya|sia|sha|po|kc|kasi|ganun|ganoon|ganon|gaano|gaanong|ganung|masyado|masyadong|that|so|very|naman|talaga|look|looks)\s+)"


def _compile(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(rf"(?<!\w)(?:{pattern})", re.IGNORECASE) for pattern in patterns)


NOT_AS_DESCRIBED_PATTERNS = _compile(
    (
        # "Fake ung 20000mah", "peke ung nakalagay na 10400mah"
        rf"(?:fake|peke)\s+{_FILL}{{0,3}}{_NUM}\s*{_UNIT}(?!\w)",
        # "hindi legit na 20k mah", "not even 2 liter", "hindi rin saktong 5m", "hindi totoong 10 meters"
        rf"{_NOT}\s+{_FILL}{{0,2}}(?:legit|totoo|totoong|true|accurate|originally|saktong|even)\s+{_FILL}{{0,3}}{_NUM}",
        # "hindi sya 10meters", "d sya totally 512gb"
        rf"{_NOT}\s+{_SOFT}{{0,3}}{_NUM}\s*{_UNIT}(?!\w)",
        # "only 30cm", "i received just 22.5w"
        rf"(?:only|just)\s+{_NUM}\s*{_UNIT}(?!\w)",
        # "sabi 10 meters", "nakalagay 30m", "nasa description niyo 2l", "akala ko ba 10meter", "65w daw"
        rf"(?:sabi|nakalagay|naka lagay|naka post|nakapost|description|discription|stated|advertised|akala ko|kala ko|akala q)\s+{_FILL}{{0,3}}{_NUM}\s*{_UNIT}(?!\w)",
        rf"{_NUM}\s*{_UNIT}\s+daw(?!\w)",
        rf"instead (?:of |a )?{_NUM}\s*{_UNIT}(?!\w)",
        r"(?:misleading|nakaka ?misleading|nakakamisleading|false advertis\w*|fake (?:description|ads?|advertisement|specs?)"
        r"|wrong (?:description|discription|specs?)|expectation vs\.? reality|not what you see)(?!\w)",
        r"(?:hindi|hinde|di|nd|ndi)\s+(?:po\s+|naman\s+|nga\s+)?(?:totoo|totoong)(?!\w)",
        # "not the same as described", "different from what is advertised", "lighter than advertised"
        rf"(?:not|hindi|di)\s+{_FILL}{{0,4}}(?:as|sa)\s+(?:described|advertised|stated)(?!\w)",
        r"(?:than|from what is|kesa sa|kaysa sa|sa)\s+(?:advertised|stated|described)(?!\w)",
        r"not (?:the )?same (?:as |with |sa |in )?(?:the |sa )?(?:description|described|pictures?|pics?|photos?|ads?)(?!\w)",
        r"different from (?:what is |the )?(?:advertised|description|pictures?)(?!\w)",
        r"kulang (?:sa|ang|yung|ung|ng)\s+(?:sukat|measurement|haba)(?!\w)",
        rf"{_NOT}\s+{_SOFT}{{0,2}}accurate\s+{_FILL}{{0,2}}(?:sizes?|sukat|measurement|medium|large)(?!\w)",
        rf"(?:sizes?|sukat|measurement)\s+{_FILL}{{0,3}}{_NOT}\s+{_SOFT}{{0,2}}accurate(?!\w)",
        # "hindi naman fast charge" (normalized "charging"), "not waterproof", "hindi photochromic"
        rf"{_NOT}\s+{_SOFT}{{0,3}}(?:fast charge\w*|fastcharg\w*|water ?proof|photochromic|smudge ?proof|true to size|5g|sensor)(?!\w)",
        # "ibang iba sa picture", "ang layo ng kulay sa picture", "malayo yung shade kesa sa nasa pic"
        rf"(?:ibang ibang?|iba|ibang|malayo|anlayo|layo|layu)\s+{_FILL}{{0,5}}(?:kesa sa|kaysa sa|sa)\s+{_FILL}{{0,2}}{_SHOWN}(?!\w)",
        # "hindi katulad nung nasa picture", "looks nothing like in the picture"
        rf"{_NOT}\s+{_FILL}{{0,3}}(?:tulad|katulad|gaya|kagaya|pareho|kapareho|same|like)\s+{_FILL}{{0,4}}{_SHOWN}(?!\w)",
        rf"nothing like\s+{_FILL}{{0,2}}{_SHOWN}(?!\w)",
        # "hindi naman ganyan yung kulay at design na nasa picture"
        rf"{_NOT}\s+{_FILL}{{0,2}}(?:ganyan|ganito|ganun|ganoon)\s+{_FILL}{{0,6}}{_SHOWN}(?!\w)",
        # "sa picture lang maganda", "maganda lng sya sa pic"
        rf"sa\s+{_SHOWN}\s+(?:lang|lng)\s+(?:naman\s+)?(?:maganda|ganda)(?!\w)",
        rf"(?:maganda|ganda)\s+{_FILL}{{0,2}}(?:lang|lng)\s+{_FILL}{{0,2}}sa\s+{_SHOWN}(?!\w)",
        rf"{_SHOWN}\s+{_FILL}{{0,5}}(?:is|are)\s+different(?!\w)",
        rf"different\s+{_FILL}{{0,4}}(?:on|in|sa|from)\s+{_FILL}{{0,2}}{_SHOWN}(?!\w)",
    )
)
# "N unit lang" needs two guards, so it is kept apart: the number is often the
# reviewer's own phone ("3500mah lang battery ng cp ko").
# Litres are left out: "1l lang ang dumating" is nearly always a wrong variant sent.
_SPEC_UNIT = r"(?:k\s*)?(?:mah|gb|tb|mbps|watts?|w|meters?|mtrs?|m|cm|inch(?:es)?|ribs)"
ONLY_NUMBER = re.compile(rf"(?<!\w){_NUM}\s*{_SPEC_UNIT}\s+(?:lang|lng|only)(?!\w)", re.IGNORECASE)
OWN_DEVICE_AFTER = re.compile(
    r"\s*(?:[\w']+\s+){0,4}(?:cp|phone|cellphone|battery ng|samsung|iphone|oppo|vivo|realme|capacity nitong|sana)(?!\w)",
    re.IGNORECASE,
)
# "yung isa kong gamit 20W lang": the reviewer's other or older device, named
# before the number (seen on a whole listing, 8 Oct 2026; no labelled round
# has a report that this guard removes).
OWN_DEVICE_BEFORE = re.compile(
    r"(?<!\w)(?:(?:yung|ung|ang|my)\s+(?:isa|luma|dati|other|old)|(?:isa|luma|dati|gamit)\s+ko(?:ng)?)\s+(?:[\w']+\s+){0,3}$",
    re.IGNORECASE,
)
CONCESSION_BEFORE = re.compile(r"(?<!\w)(?:kahit|may|pang)\s+$", re.IGNORECASE)
# "mas light compared dun sa picture" is a report; "mas maganda kesa sa picture" is praise.
COMPARED_TO_SHOWN = re.compile(rf"(?<!\w)(?:compared?|kumpara|kesa|kaysa)\s+{_FILL}{{0,3}}{_SHOWN}(?!\w)", re.IGNORECASE)
PRAISE_BEFORE = re.compile(r"(?<!\w)(?:mas\s+)?(?:maganda|ganda|okay|ok|better|nice)\s+(?:[\w']+\s+){0,3}$", re.IGNORECASE)
SPECULATION = re.compile(r"(?<!\w)(?:baka|bka|maybe|might)\s+(?:[\w']+\s+){0,2}$", re.IGNORECASE)

QUALITY_PATTERNS = _compile(
    (
        # weak
        r"(?:(?:ma|napaka)?hina|mahinang)(?!\w)",
        rf"{_NOT_WIDE}\s+{_DEGREE}{{0,2}}(?:ka)?(?:lakas|malakas|kalakasan|loud)(?!\w)",
        r"(?:volume too low|mas tumutunog pa)(?!\w)",
        rf"mas malakas pa\s+{_FILL}{{0,5}}(?:kesa|kaysa)(?!\w)",
        r"mas malakas pa (?:ang )?(?:ihip|hininga)(?!\w)",
        # sound
        rf"sabog\s+{_FILL}{{0,2}}(?:tunog|sounds?|speaker|mic|microphone)(?!\w)",
        rf"(?:tunog|sounds?|speaker|mic|microphone)\s+{_FILL}{{0,3}}sabog(?!\w)",
        rf"basag\s+{_FILL}{{0,2}}(?:tunog|sounds?|music|pakinggan)(?!\w)",
        rf"(?:tunog|sounds?)\s+{_FILL}{{0,2}}basag(?!\w)",
        r"(?:sabog sya|tunog lata|wasak sound|maugong|garalgal|not enough bass|bass is meh)(?!\w)",
        rf"(?:pangit|panget)\s+{_FILL}{{0,2}}(?:tunog|sounds?|boses)(?!\w)",
        rf"{_NOT_WIDE}\s+{_FILL}{{0,2}}(?:maganda|kaganda|ganda)\s+{_FILL}{{0,3}}(?:tunog|sounds?)(?!\w)",
        # quality words
        r"(?:low|poor|bad|cheap|awful) (?:sound |build |product )?quality(?!\w)",
        r"quality\s*:?\s*(?:is |was |nya |niya |ng \w+ )?(?:not good|awful|bad|poor|pangit|panget|low)(?!\w)",
        rf"(?:pangit|panget|baba|bulok)\s+{_FILL}{{0,2}}quality(?!\w)",
        rf"{_NOT_WIDE}\s+{_FILL}{{0,2}}(?:maganda|ganda|kaganda|good|high)\s+{_FILL}{{0,3}}quality(?!\w)",
        r"(?:low class|poor product|poor performance|poor performace|cheap plastic|mumurahin|substandard|cheaply made"
        r"|cheap material|flimsy|meh lang)(?!\w)",
        r"functionality\s*:?\s*(?:poor|pangit|panget)(?!\w)",
        # thin, fragile, toy-like
        r"(?:ma|ka|napaka)?nipis\w*",
        r"marupok(?!\w)",
        r"(?:madali|madaling|mabilis|dali|bilis)\s+(?:(?:lang|lng|na|ma|sya|siya|nya|nga)\s+){0,2}"
        r"(?:sira|masira|mapunit|punit|maputol|putol|matanggal|tanggal|mabura|mawala|kumupas|lumuwag|maluwag)(?!\w)",
        rf"{_NOT_WIDE}\s+{_DEGREE}{{0,3}}(?:ka)?(?:tibay|matibay|sturdy|durable|stable)(?!\w)",
        r"(?:parang|para siyang|para syang|para lang|mukhang|mukang)\s+(?:laruan|laroan)(?!\w)",
        # does not hold or fit
        r"(?:nahuhulog|nalalaglag|lumalaylay|wobbly|super loose|sobrang luwag)(?!\w)",
        rf"(?:maluwag|maluwang|loose)\s+{_FILL}{{0,3}}(?:turnilyo|screws?|pagkakabit|rj45|saksak\w*|plug)(?!\w)",
        rf"(?:turnilyo|screws?|saksak\w*|plug)\s+{_FILL}{{0,3}}(?:maluwag|maluwang|loose)(?!\w)",
        rf"{_NOT_WIDE}\s+{_DEGREE}{{0,2}}(?:madikit|dumidikit|nakadikit|mahigpit|pantay|pare pareho|pareho|parehas|straight)(?!\w)",
        r"(?:hindi|hinde|di|d|nd|ndi)\s+(?:(?:rin|din|po|p|masyadong)\s+){0,2}maayos\s+(?:(?:ang|yung|ung)\s+)?"
        r"(?:tahi|pagka ?tahi|pagkakabit|pagkakadikit|pgkakadikit)(?!\w)",
        # does not last (scent, colour)
        rf"(?:not long lasting|{_NOT_WIDE}\s+(?:(?:sya|siya|naman)\s+)?(?:long lasting|naglalast|nagtatagal))(?!\w)",
        # Second revision (8 Oct 2026): written from two thirds of the labelled
        # sentences and checked on the third (research/improve_label_rules.py).
        # noisy
        r"(?:maingay|ang ingay|sobrang ingay|napaka ?ingay)(?!\w)",
        # lag in sound or response
        rf"delay(?:ed)?\s+{_FILL}{{0,2}}(?:mic|sound|tunog|audio|response|pagsulat)(?!\w)",
        rf"(?:mic|sound|tunog|audio|feature)\s+{_FILL}{{0,2}}delay(?:ed)?(?!\w)",
        # "material not that expected". "The item is not good" is left out: on the
        # sentences read it was as often a fault as poor quality.
        r"(?:material|materials)\s+(?:is |was |are )?not\s+(?:very |so |that |really )?(?:good|great|expected)(?!\w)",
        # "panget gamitin", "pangit ang pagkakagawa"
        rf"(?:pangit|panget)\s+{_FILL}{{0,2}}(?:gamitin|pagkakagawa|pagkagawa|gawa|materials?|tela|tahi|print|stiker|sticker)(?!\w)",
        r"(?:napaka ?low|pucho pucho|puchu puchu)\s+quality(?!\w)",
        # "palpak ang power bank ninyo". "Bulok ang item" is left out: usually said of something broken.
        rf"palpak\s+{_FILL}{{0,2}}(?:item|product|power ?bank|unit|gawa|binigay)(?!\w)",
        # looks used
        r"(?:parang|mukhang|mukang)\s+(?:gamit na|ginamit|luma|second ?hand|used)(?!\w)",
        r"(?:not brand new|second ?hand)(?!\w)",
        # "medyo basag sya"
        r"(?:medyo|mejo|medjo)\s+basag(?!\w)",
        r"mas malakas pa\s+(?:ang\s+)?(?:sound|tunog|volume|speaker)\s+ng\s+(?:cp|phone|cellphone)(?!\w)",
        # stiff or hard to use
        r"(?:sobrang|masyadong|napaka)\s*tigas(?!\w)",
        r"hard (?:to )?press(?!\w)",
        r"hirap\s+(?:pa\s+)?(?:pindutin|buksan|hilahin|isaksak|ikabit|i ?connect|iapply)(?!\w)",
        r"(?:wonky|difficult to connect)(?!\w)",
        # too light, all plastic
        r"masyado(?:ng)?\s+magaan(?!\w)",
        r"(?:super|puro)\s+(?:fake\s+)?plastic(?!\w)",
        # cannot carry the weight
        rf"{_NOT_WIDE}\s+(?:nya\s+|niya\s+)?(?:kinakaya|kaya)\s+(?:yung|ung|ang)\s+bigat(?!\w)",
        r"malabo\s+(?:ang\s+|yung\s+|ung\s+)?(?:print|screen|camera|video|tunog|sound|mic)(?!\w)",
        r"expected ko\s+(?:makapal|matibay|malakas)(?!\w)",
    )
)
# Damage that leaves the product usable. Counted only when it is on the
# product, not on the box.
COSMETIC = re.compile(
    r"(?<!\w)(?:may (?:konti(?:ng)?|kunti(?:ng)?|maliit na) (?:basag|sira|damage|tama|crack)|dents?|dented|gasgas|scratch(?:es|ed)?|yupi|nayupi|may (?:mga )?butas|punit|tastas|cracks?|cracked|tuklap"
    r"|ukab|kalawang|tabingi|baliko|stains?|mantsa|dirty|dusty|madumi|marumi|kupas|nakaangat|umangat|nakabuka|meron(?:g)? butas)(?!\w)",
    re.IGNORECASE,
)
PACKAGING = frozenset(
    {"box", "boxes", "kahon", "karton", "package", "packaging", "packing", "parcel", "balot", "pagkakabalot",
     "wrapper", "wrapping", "pouch", "plastic", "bubble"}
)
# The shared negation test does not see through "wala pong gasgas"; the two
# new labels use this wider one.
NEGATED_BEFORE = re.compile(
    r"(?<!\w)(?:wala|walang|hindi|hinde|di|d|nd|ndi|no|not|without|zero|never)\s+"
    r"(?:(?:po|pong|naman|namn|nman|namang|ng|man|talaga|any|kahit|ka|ko|sya|siya|masyado|masyadong|ganun|gaano"
    r"|gaanong|yung|ung|tipong|scratches|dents?|gasgas|or|o|at)\s+){0,3}$",
    re.IGNORECASE,
)
# "tunog h di basag": a negator inside the matched words.
NEGATED_INSIDE = re.compile(
    r"(?<!\w)(?:hindi|hinde|di|d|nd|ndi|not|wala|walang)\s+(?:naman\s+|po\s+)?(?:basag|sabog|mahina|hina|manipis|nipis)(?!\w)",
    re.IGNORECASE,
)
# "may nababasa ako sa comments na mabilis nasira": someone else's report.
HEARSAY_BEFORE = re.compile(
    r"(?<!\w)(?:nababasa|nabasa|comments|reviews|sabi nila|sabi ng iba)\s+(?:[\w']+\s+){0,5}$", re.IGNORECASE
)


def not_as_described_spans(normalized: str) -> list[tuple[int, int]]:
    """Where the reviewer disputes a number, a feature or the photos."""
    spans = [
        found.span()
        for pattern in NOT_AS_DESCRIBED_PATTERNS
        for found in pattern.finditer(normalized)
        if not SPECULATION.search(normalized[: found.start()]) and "dual shock" not in found.group(0).casefold()
    ]
    for found in ONLY_NUMBER.finditer(normalized):
        before = normalized[: found.start()]
        if (
            OWN_DEVICE_AFTER.match(normalized[found.end():])
            or CONCESSION_BEFORE.search(before)
            or OWN_DEVICE_BEFORE.search(before)
        ):
            continue
        spans.append(found.span())
    for found in COMPARED_TO_SHOWN.finditer(normalized):
        if not PRAISE_BEFORE.search(normalized[: found.start()]):
            spans.append(found.span())
    return sorted(set(spans))


# "pag mahina lang ang volume": a condition, not a complaint.
CONDITION_BEFORE = re.compile(r"(?<!\w)(?:pag|kapag|kung|if|when)\s+(?:[\w']+\s+){0,1}$", re.IGNORECASE)
# "ang hina na nya mag charge": slow charging is a defect (guide 1.3), not poor quality.
CHARGING_AFTER = re.compile(r"\s*(?:[\w']+\s+){0,3}(?:mag ?charge|magcharge|charge)(?!\w)", re.IGNORECASE)


def _on_the_box(normalized: str, span: tuple[int, int], reach: int = 5) -> bool:
    before = re.findall(r"[\w']+", normalized[: span[0]])[-reach:]
    after = re.findall(r"[\w']+", normalized[span[1]:])[:reach]
    return any(word in PACKAGING for word in before + after)


def quality_spans(normalized: str) -> list[tuple[int, int]]:
    """Where the reviewer says the product is weak, cheap, flimsy or marked."""
    spans = [
        found.span()
        for pattern in QUALITY_PATTERNS
        for found in pattern.finditer(normalized)
        if not NEGATED_BEFORE.search(normalized[: found.start()])
        and not NEGATED_INSIDE.search(found.group(0))
        and not HEARSAY_BEFORE.search(normalized[: found.start()])
        and not CONDITION_BEFORE.search(normalized[: found.start()])
        and not ("hina" in found.group(0).casefold() and CHARGING_AFTER.match(normalized[found.end():]))
    ]
    spans += [
        found.span()
        for found in COSMETIC.finditer(normalized)
        if not NEGATED_BEFORE.search(normalized[: found.start()]) and not _on_the_box(normalized, found.span())
    ]
    return sorted(set(spans))
