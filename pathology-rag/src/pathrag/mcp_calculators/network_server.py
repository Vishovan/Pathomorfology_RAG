from typing import Annotated

from mcp.server.mcpserver import MCPServer
from pydantic import Field

server = MCPServer("pathology-catalog")
Grade = Annotated[int, Field(strict=True, ge=1, le=3)]
Pattern = Annotated[int, Field(strict=True, ge=3, le=5)]
Query = Annotated[str, Field(min_length=1, max_length=500)]


@server.tool()
def gleason(primary: Pattern, secondary: Pattern) -> dict:
    """Рак простаты · Gleason Grade Group по двум установленным паттернам 3–5."""
    score = primary + secondary
    group = {6: 1, 7: 2 if primary == 3 else 3, 8: 4, 9: 5, 10: 5}[score]
    return {"score": score, "group": group, "source": "https://www.cancer.gov/types/prostate/hp/prostate-treatment-pdq"}


@server.tool()
def fnclcc(differentiation: Grade, mitoses: Grade, necrosis: Annotated[int, Field(strict=True, ge=0, le=2)]) -> dict:
    """Саркомы мягких тканей · FNCLCC по выставленным баллам дифференцировки, митозов и некроза."""
    score = differentiation + mitoses + necrosis
    return {"score": score, "grade": 1 + int(score >= 4) + int(score >= 6),
            "source": "https://pubmed.ncbi.nlm.nih.gov/6693192/"}


@server.tool()
def allred(proportion: Annotated[int, Field(strict=True, ge=0, le=5)],
           intensity: Annotated[int, Field(strict=True, ge=0, le=3)]) -> dict:
    """Рак молочной железы · Allred по двум выставленным баллам доли и интенсивности окрашивания."""
    if (proportion == 0) != (intensity == 0):
        raise ValueError("При отсутствии окрашивания оба компонента должны равняться нулю.")
    return {"score": proportion + intensity, "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5681416/"}


@server.tool()
def search_images(query: Query) -> list[dict]:
    """Атлас · Поиск микропрепаратов по заболеванию или окраске."""
    from pathrag.retrieval.image_retriever import search_pathology_images

    return search_pathology_images(query)


@server.tool()
def search_corpus(query: Query) -> list[dict]:
    """База знаний · Поиск материалов по нозологии."""
    from pathrag.retrieval.hybrid_retriever import retrieve

    return retrieve(query)


if __name__ == "__main__":
    server.run(transport="sse", host="127.0.0.1", port=8000)
