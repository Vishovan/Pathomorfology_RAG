"""Эвристики: из подписи к рисунку достаём увеличение, окраску, орган и диагноз (на ru и en)."""
import re

_MAG_OK = {4, 5, 10, 20, 25, 40, 50, 60, 63, 100, 125, 150, 200, 250, 300, 400, 600, 630, 1000}
_OBJ = {4, 10, 20, 40, 60, 100}

_MAG_PATTERNS = [
    re.compile(r"(?<![A-Za-zА-Яа-яЁё])(?:[xх×]|ув(?:еличение|\.)?\s*)\s*(\d{1,4})\b", re.I),
    re.compile(r"\b(\d{1,4})\s*[xх×](?![A-Za-zА-Яа-яЁё])", re.I),
]
_OCULAR_OBJ = re.compile(r"\b(\d{1,2})\s*[xх×]\s*(\d{1,3})\b", re.I)

_STAINS = [
    ("H&E", r"гематоксилин\w*[\s\-]*(?:и[\s\-]*)?эозин|\bГ\s?[-–]?\s?Э\b|H\s?&\s?E|hematoxylin"),
    ("PAS", r"\bPAS\b|\bШИК\b"),
    ("Van Gieson", r"ван[\s\-]*гизон|van\s*gieson"),
    ("Masson trichrome", r"массон|masson|trichrome|трихром"),
    ("Congo red", r"конго\s*красн|congo\s*red"),
    ("Sudan III", r"судан|sudan"),
    ("Perls", r"перлс|perls|prussian"),
    ("Mallory", r"малори|mallory"),
    ("Giemsa", r"гимз|giemsa"),
    ("Ziehl-Neelsen", r"цил\w*[\s\-]*нильсен|ziehl"),
    ("Alcian blue", r"альциан|alcian"),
    ("Silver impregnation", r"серебр|silver|импрегнац"),
    ("IHC", r"иммуногистохим|immunohisto|\bIHC\b"),
]
_STAINS = [(n, re.compile(p, re.I)) for n, p in _STAINS]

_ORGANS = [
    ("легкие", r"легк|lung|pulmon|пневмон|pneumon|бронх|bronch|альвеол|alveol"), ("печень", r"печен|liver|hepat"),
    ("почки", r"почк|kidney|renal|нефр|nephr|гломерул|glomerul"), ("сердце", r"сердц|миокард|heart|myocard|cardiac"),
    ("головной мозг", r"головн\w*\s+мозг|\bмозг|brain|cerebr"), ("селезенка", r"селезен|spleen"),
    ("лимфатический узел", r"лимфатическ\w*\s+узел|лимфоузл|lymph\s*node"),
    ("желудок", r"желуд|stomach|gastric"), ("кишечник", r"кишк|кишечн|colon|intestin|bowel"),
    ("поджелудочная железа", r"поджелудочн|pancrea"), ("щитовидная железа", r"щитовидн|thyroid"),
    ("надпочечник", r"надпочечн|adrenal"), ("молочная железа", r"молочн\w*\s+желез|breast|mammary"),
    ("матка", r"\bматк|uterus|endometri|cervix|шейк\w*\s+матк"), ("яичник", r"яичник|ovary|ovarian"),
    ("предстательная железа", r"предстательн|простат|prostate"), ("кожа", r"\bкож[аеуиыо]|skin|cutaneous"),
    ("кость", r"\bкост|bone|osseous"), ("мышца", r"мышц|muscle"),
    ("мочевой пузырь", r"мочев\w*\s+пузыр|bladder"), ("пищевод", r"пищевод|esophag"),
    ("сосуды", r"аорт|aort|артери|arter|сосуд|vessel"),
]
_ORGANS = [(n, re.compile(p, re.I)) for n, p in _ORGANS]

_FIG_PREFIX = re.compile(
    r"^\s*(?:рис(?:унок)?|fig(?:ure)?)\.?\s*\d+(?:[.\-]\d+)*[а-яa-z]?\s*[.:–—\-]?\s*", re.I
)
_TAIL = re.compile(r"\s*[,;(]\s*(?:окраск|ув\b|увелич|stain|[xх×]\s?\d|\d+\s?[xх×])", re.I)


def extract_magnification(text: str) -> str | None:
    if not text:
        return None
    m = _OCULAR_OBJ.search(text)
    if m:
        oc, ob = int(m.group(1)), int(m.group(2))
        if oc in (7, 10, 15) and ob in _OBJ and oc * ob in _MAG_OK:
            return f"{oc * ob}x"
    for pat in _MAG_PATTERNS:
        for m in pat.finditer(text):
            v = int(m.group(1))
            if v in _MAG_OK:
                return f"{v}x"
    return None


def extract_stain(text: str) -> str | None:
    best = None
    for name, pat in _STAINS:
        m = pat.search(text or "")
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), name)
    return best[1] if best else None


def extract_organ(text: str) -> str | None:
    best = None
    for name, pat in _ORGANS:
        m = pat.search(text or "")
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), name)
    return best[1] if best else None


def extract_diagnosis(caption: str) -> str | None:
    """Первое предложение подписи без «Рис. N» и хвоста про окраску/увеличение. Эвристика."""
    c = _FIG_PREFIX.sub("", (caption or "").strip())
    first = re.split(r"(?<=[.!?])\s+|\n", c, maxsplit=1)[0].strip(" .;:–—-")
    first = _TAIL.split(first, maxsplit=1)[0].strip(" .;:–—-,")
    for _, pat in _STAINS:  # «Диагноз, H&E» -> «Диагноз»
        m = pat.search(first)
        if m and m.start() >= 4 and first[:m.start()].rstrip().endswith((",", ";", "(", ":", "-", "–")):
            first = first[:m.start()].strip(" .;:–—-,(")
            break
    if re.match(r"^(окраск|ув\b|увелич|stain|magnif)", first, re.I):
        return None
    return first if 3 <= len(first) <= 120 else None
