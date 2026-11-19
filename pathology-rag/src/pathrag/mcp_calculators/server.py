"""Запуск: python -m pathrag.mcp_calculators.server."""
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from . import calculators

server = MCPServer("pathology-calculators")
Grade = Annotated[int, Field(strict=True, ge=1, le=3)]
Count = Annotated[int, Field(strict=True, ge=0)]
Total = Annotated[int, Field(strict=True, gt=0)]
Percent = Annotated[float, Field(strict=True, ge=0, le=100)]


@server.tool()
def nottingham(tubules: Grade, nuclei: Grade, mitoses: Grade) -> dict:
    """Nottingham Grade. Введите баллы за образование трубочек, вид ядер и митозы."""
    return calculators.nottingham(tubules, nuclei, mitoses)


@server.tool()
def ki67(positive: Count, total: Total) -> dict:
    """Индекс Ki-67. Введите число положительных и общее число подсчитанных опухолевых клеток."""
    return calculators.ki67(positive, total)


@server.tool()
def h_score(negative: Percent, weak: Percent, moderate: Percent, strong: Percent) -> dict:
    """H-score. Введите проценты клеток для четырёх интенсивностей окрашивания по порядку."""
    return calculators.h_score(negative, weak, moderate, strong)


@server.tool()
def list_calculators() -> dict:
    """Методики, единицы измерения и ограничения на аргументы."""
    return {
        "nottingham": {"description": calculators.nottingham.__doc__, "unit": "баллы",
                       "limits": "Каждый компонент 1–3. Сумма 3–9."},
        "ki67": {"description": calculators.ki67.__doc__, "unit": "клетки",
                 "limits": "Целые числа, 0 ≤ positive ≤ total, total > 0."},
        "h_score": {"description": calculators.h_score.__doc__, "unit": "%",
                    "limits": "Каждая доля 0–100%, сумма 100%."},
    }


if __name__ == "__main__":
    server.run(transport="stdio")
