import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathrag.retrieval.legal_kb import get_medical_device_boundary_info, search_gost_standards


def test_search_gost_34_602():
    results = search_gost_standards("ГОСТ 34.602-2020 техническое задание")
    assert len(results) > 0
    top = results[0]
    assert "34.602" in top.get("gost", "")


def test_search_safe_development():
    results = search_gost_standards("ГОСТ Р 56939 безопасная разработка")
    assert len(results) > 0
    assert any("56939" in r.get("gost", "") for r in results)


def test_medical_device_boundary():
    boundary = get_medical_device_boundary_info()
    assert "pathology_rag_strategy" in boundary
    assert "is_not_samd" in boundary
    assert len(boundary["is_not_samd"]) > 0
