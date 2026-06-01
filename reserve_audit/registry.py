"""Target registry enforcing comparable recoverability rankings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class TargetNode:
    """One recoverability audit node."""

    node_id: str
    modality: str
    encoder: str
    target_name: str
    target_kind: str
    target_entropy: float | None
    baseline_vars: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TargetRegistry:
    """Small schema plus guardrails for target-separated comparisons."""

    def __init__(self, nodes: Iterable[TargetNode | dict[str, Any]] | None = None) -> None:
        self._nodes: dict[str, TargetNode] = {}
        for node in nodes or []:
            self.add(node)

    def add(self, node: TargetNode | dict[str, Any]) -> None:
        item = node if isinstance(node, TargetNode) else TargetNode(**node)
        if item.node_id in self._nodes:
            raise ValueError(f"duplicate registry node_id: {item.node_id}")
        if not item.target_kind:
            raise ValueError("target_kind is required")
        if not item.target_name:
            raise ValueError("target_name is required")
        self._nodes[item.node_id] = item

    def get(self, node_id: str) -> TargetNode:
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise KeyError(f"unknown registry node_id: {node_id}") from exc

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {key: node.to_dict() for key, node in self._nodes.items()}

    def assert_comparable(self, node_ids: Iterable[str]) -> str:
        nodes = [self.get(node_id) for node_id in node_ids]
        if not nodes:
            raise ValueError("at least one node is required")
        kinds = {node.target_kind for node in nodes}
        if len(kinds) != 1:
            details = {node.node_id: node.target_kind for node in nodes}
            raise ValueError(
                "registry target-separation violation: one ranking cannot mix different target_kind values; "
                f"got {details}"
            )
        return next(iter(kinds))

    def ranking_table(self, rows: list[dict[str, Any]], node_key: str = "node_id") -> list[dict[str, Any]]:
        """Validate and return rows sorted by decreasing recoverability.

        Rows must include ``node_id`` and either ``fraction_of_entropy`` or
        ``score``. The registry refuses to rank mixed target kinds.
        """

        node_ids = [str(row[node_key]) for row in rows]
        self.assert_comparable(node_ids)

        def score(row: dict[str, Any]) -> float:
            if "fraction_of_entropy" in row:
                return float(row["fraction_of_entropy"])
            return float(row["score"])

        return sorted(rows, key=score, reverse=True)
