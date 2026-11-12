from pathrag.agent import ask_agent


def test_agent_breast_cancer_flow():
    res = ask_agent("Какие критерии оценки HER2 2+ при раке молочной железы?")
    assert res.get("intent") == "breast_cancer_ihc"
    assert "HER2" in res.get("final_answer", "")
    assert len(res.get("citations", [])) > 0
    assert "Справочно-образовательная система" in res.get("disclaimer", "")


def test_agent_legal_flow():
    res = ask_agent("Какие процедуры пройти в Роспатенте и Росздравнадзоре?")
    assert res.get("intent") == "legal_gost"
    assert "Роспатент" in res.get("final_answer", "")
    assert "Росздравнадзор" in res.get("final_answer", "")
    assert "дисклеймер" in res.get("disclaimer", "").lower() or "справка" in res.get("disclaimer", "").lower()


def test_agent_image_flow():
    res = ask_agent("Покажи микропрепарат серого опеченения при пневмонии")
    assert res.get("intent") == "image_search"
    assert len(res.get("retrieved_images", [])) > 0
    img = res["retrieved_images"][0]
    assert "серого опеченения" in img.get("title", "").lower() or "пневмония" in img.get("title", "").lower()
