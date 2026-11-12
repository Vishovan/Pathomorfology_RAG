import shutil
from pathlib import Path

from pathrag import imeta
from pathrag.images import make_record
from pathrag.web_images import extract_images


def test_magnification():
    assert imeta.extract_magnification("Окраска гематоксилин-эозин, ув. 200") == "200x"
    assert imeta.extract_magnification("H&E, x400") == "400x"
    assert imeta.extract_magnification("H&E 100X") == "100x"
    assert imeta.extract_magnification("окр. ГЭ, х 400") == "400x"
    assert imeta.extract_magnification("увеличение 10×40") == "400x"
    assert imeta.extract_magnification("Рисунок 12. в 2024 году") is None
    assert imeta.extract_magnification("") is None


def test_stain_organ():
    assert imeta.extract_stain("Окраска гематоксилином и эозином") == "H&E"
    assert imeta.extract_stain("Окраска гематоксилин-эозин") == "H&E"
    assert imeta.extract_stain("Stain: PAS reaction") == "PAS"
    assert imeta.extract_stain("по Ван Гизону") == "Van Gieson"
    assert imeta.extract_stain("без окраски") is None
    assert imeta.extract_organ("Крупозная пневмония, легкое") == "легкие"
    assert imeta.extract_organ("Cirrhosis of the liver") == "печень"


def test_diagnosis():
    d = imeta.extract_diagnosis("Рис. 12. Крупозная пневмония. Окраска гематоксилин-эозин, ув. 200.")
    assert d == "Крупозная пневмония"
    assert imeta.extract_diagnosis("Окраска гематоксилин-эозин, ув. 200") is None
    assert imeta.extract_diagnosis("Рисунок 3") is None
    assert imeta.extract_diagnosis("Hepatocellular carcinoma, H&E, x200") == "Hepatocellular carcinoma"


def test_make_record():
    r = make_record(source_id="s", doc_key="k", idx=1, default_lang="ru", section_path="Глава 5 > Воспаление",
                    nosology="Воспаление", caption="Рис. 12. Крупозная пневмония. Окраска гематоксилин-эозин, ув. 200.")
    assert r["diagnosis"] == "Крупозная пневмония" and r["diagnosis_source"] == "caption"
    assert r["stain"] == "H&E" and r["magnification"] == "200x" and r["organ"] == "легкие"
    assert "Крупозная пневмония" in r["search_text"] and r["language"] == "ru"
    # без подписи диагноз берётся из заголовка
    r2 = make_record(source_id="s", doc_key="k", idx=2, default_lang="ru", section_path="Глава 5", nosology="Фибринозное воспаление", caption="")
    assert r2["diagnosis"] == "Фибринозное воспаление" and r2["caption_source"] == "heading"
    # нет ни подписи, ни контекста: пропуск
    assert make_record(source_id="s", doc_key="k", idx=3, default_lang="ru", caption="") is None
    # ручные значения важнее
    r3 = make_record(source_id="s", doc_key="k", idx=4, default_lang="en", caption="Liver", overrides={"stain": "PAS"})
    assert r3["stain"] == "PAS"
    # id детерминирован
    assert r["image_id"] == make_record(source_id="s", doc_key="k", idx=1, default_lang="ru", caption="x")["image_id"]


HTML = """<html><title>Atlas</title><body><nav><img src='/nav.png' alt='navigation image here'></nav><main>
<h2>Крупозная пневмония</h2>
<figure><img src='/img/a.jpg' width='400'><figcaption>Серое опеченение, ув. 100</figcaption></figure>
<table><tr><td><img src='/img/b.jpg' alt='Фибринозный плеврит, H&E'></td></tr></table>
<div><img src='/img/c.jpg'><p>Красное опеченение легкого, окраска ГЭ</p></div>
<img src='/img/logo.png'><img src='/img/tiny.jpg' width='20'><img src='data:image/png;base64,AAA'>
</main></body></html>"""


def test_extract_images():
    title, imgs = extract_images(HTML, "https://x.org/atlas/page.html")
    urls = [i["image_url"] for i in imgs]
    assert urls == ["https://x.org/img/a.jpg", "https://x.org/img/b.jpg", "https://x.org/img/c.jpg"]
    assert imgs[0]["caption"].startswith("Серое опеченение") and imgs[0]["caption_how"] == "figcaption"
    assert imgs[1]["caption_how"] == "alt"
    assert "Красное опеченение" in imgs[2]["caption"]
    assert imgs[0]["heading"] == "Крупозная пневмония"


def test_pdf_figures_and_wsi_csv_dry_run():
    from PIL import Image

    from pathrag import pipeline
    from pathrag.config import SETTINGS
    from pathrag.parse_pdf import Figure

    figs = [Figure(1, 5, "Рис. 1. Крупозная пневмония. ГЭ, ув. 200", "Глава 5 > Воспаление", "Воспаление", Image.new("RGB", (900, 700), "pink")),
            Figure(2, 6, "", None, None, Image.new("RGB", (300, 300), "blue"))]
    recs = pipeline.figures_to_records({"id": "test_src", "language": "ru"}, "doc123", "data/raw/textbooks/t.pdf", Path("t.pdf"), figs)
    try:
        assert len(recs) == 1 and recs[0]["page"] == 5
        thumb = Path(__file__).resolve().parents[1] / recs[0]["thumb_path"]
        assert thumb.exists() and max(Image.open(thumb).size) <= SETTINGS.thumb_max_px
    finally:
        shutil.rmtree(SETTINGS.images_dir / "test_src", ignore_errors=True)
