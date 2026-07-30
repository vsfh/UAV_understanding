from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Event:
    name: str
    definition: str


@dataclass(frozen=True)
class Ontology:
    events: tuple[Event, ...]
    confusion_edges: tuple[tuple[str, str], ...]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(event.name for event in self.events)

    @property
    def definitions(self) -> dict[str, str]:
        return {event.name: event.definition for event in self.events}

    def neighbors(self, label: str) -> tuple[str, ...]:
        result: list[str] = []
        for left, right in self.confusion_edges:
            if left == label:
                result.append(right)
            elif right == label:
                result.append(left)
        return tuple(result)


def load_ontology(path: str | Path) -> Ontology:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    events = tuple(Event(item["name"], item["definition"]) for item in raw["events"])
    labels = [event.name for event in events]
    if len(labels) != len(set(labels)):
        raise ValueError("Ontology contains duplicate event names")

    edges = tuple(tuple(edge) for edge in raw["confusion_edges"])
    unknown = {label for edge in edges for label in edge} - set(labels)
    if unknown:
        raise ValueError(f"Confusion edges reference unknown events: {sorted(unknown)}")
    return Ontology(events, edges)


def load_label_subset(path: str | Path, ontology: Ontology) -> tuple[str, ...]:
    labels = tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )
    if len(labels) != len(set(labels)):
        raise ValueError(f"Duplicate labels in {path}")
    unknown = set(labels) - set(ontology.labels)
    if unknown:
        raise ValueError(f"Unknown labels in {path}: {sorted(unknown)}")
    return labels
