from dataclasses import dataclass


@dataclass
class Block:
    kind: str          # heading | text | table
    text: str
    level: int = 0     # для heading: 1 = самый крупный
    page: int | None = None


@dataclass
class Chunk:
    text: str
    metadata: dict
