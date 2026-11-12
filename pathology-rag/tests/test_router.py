import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathrag.agent.router import route_intent


def test_route_breast_cancer():
    assert route_intent("Какие критерии оценки HER2 2+ при РМЖ?") == "breast_cancer_ihc"
    assert route_intent("Порог Ki-67 для люминального А подтипа") == "breast_cancer_ihc"
    assert route_intent("Nottingham score grade 2 рак молочной железы") == "breast_cancer_ihc"


def test_route_legal_gost():
    assert route_intent("Как составить ТЗ по ГОСТ 34.602-2020?") == "legal_gost"
    assert route_intent("В чем отличие регистрации в Роспатенте от Росздравнадзора?") == "legal_gost"
    assert route_intent("Требования ГОСТ Р 56939-2024 по безопасной разработке") == "legal_gost"


def test_route_image_search():
    assert route_intent("Покажи микропрепарат серого опеченения при пневмонии") == "image_search"
    assert route_intent("Фото микропрепарата судан 3 стеатоз печени") == "image_search"
    assert route_intent("WSI слайд туберкулезной гранулемы") == "image_search"


def test_route_explicit_override():
    assert route_intent("любой текст", explicit_mode="legal") == "legal_gost"
    assert route_intent("любой текст", explicit_mode="images") == "image_search"
    assert route_intent("любой текст", explicit_mode="pathology") == "pathology_qa"
