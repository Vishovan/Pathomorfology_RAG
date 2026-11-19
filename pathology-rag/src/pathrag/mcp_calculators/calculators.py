from math import isclose, isfinite


def nottingham(tubules: int, nuclei: int, mitoses: int) -> dict:
    """Степень злокачественности инвазивной карциномы молочной железы по трём выставленным баллам."""
    components = {"tubules": tubules, "nuclei": nuclei, "mitoses": mitoses}
    if any(type(n) is not int or not 1 <= n <= 3 for n in components.values()):
        raise ValueError("Каждый компонент должен быть целым баллом от 1 до 3.")
    score = sum(components.values())
    return {"score": score, "grade": 1 + int(score >= 6) + int(score >= 8), "components": components,
            "source": "https://doi.org/10.1111/j.1365-2559.1991.tb00229.x"}


def ki67(positive: int, total: int) -> dict:
    """Доля Ki-67-положительных опухолевых клеток среди подсчитанных, в процентах."""
    if type(total) is not int or type(positive) is not int or total <= 0 or not 0 <= positive <= total:
        raise ValueError("Введите целые количества клеток: всего > 0 и 0 ≤ положительных ≤ всего.")
    return {"percent": round(100 * positive / total, 2), "positive": positive, "total": total,
            "source": "https://pubmed.ncbi.nlm.nih.gov/33369635/"}


def h_score(negative: float, weak: float, moderate: float, strong: float) -> dict:
    """H-score окрашивания по процентам клеток с интенсивностями 0, 1+, 2+ и 3+."""
    percentages = [negative, weak, moderate, strong]
    if any(type(p) not in (int, float) or not 0 <= p <= 100 or not isfinite(p) for p in percentages):
        raise ValueError("Каждый процент должен быть числом от 0 до 100.")
    if not isclose(sum(percentages), 100, rel_tol=0, abs_tol=1e-8):
        raise ValueError("Сумма четырёх процентов должна равняться 100.")
    contributions = [intensity * percent for intensity, percent in enumerate(percentages)]
    return {"score": round(sum(contributions), 6), "percentages": percentages, "contributions": contributions,
            "source": "https://doi.org/10.1038/s41523-026-00895-3"}
