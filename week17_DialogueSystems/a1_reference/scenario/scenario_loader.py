"""加载场景 JSON 与槽位 Excel 配置。"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd


@dataclass
class SlotConfig:
    name: str
    query: str
    values: List[str]
    is_regex: bool = False
    regex_pattern: Optional[str] = None


@dataclass
class ScenarioNode:
    id: str
    intents: List[str]
    slots: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    response: str = ""
    child_ids: List[str] = field(default_factory=list)


@dataclass
class Scenario:
    name: str
    path: Path
    nodes: Dict[str, ScenarioNode]
    entry_id: str


def _parse_slot_values(raw: str) -> tuple[List[str], bool, Optional[str]]:
    raw = str(raw).strip()
    if "|" not in raw and re.search(r"\[[^\]]+\]", raw):
        return [], True, raw
    return [v.strip() for v in raw.split("|") if v.strip()], False, None


def load_slot_config(excel_path: Union[str, Path]) -> Dict[str, SlotConfig]:
    df = pd.read_excel(excel_path, sheet_name=0)
    configs: Dict[str, SlotConfig] = {}
    for _, row in df.iterrows():
        name = str(row["slot"]).strip()
        values, is_regex, pattern = _parse_slot_values(row["values"])
        configs[name] = SlotConfig(
            name=name,
            query=str(row["query"]).strip(),
            values=values,
            is_regex=is_regex,
            regex_pattern=pattern,
        )
    return configs


def load_scenario(json_path: Union[str, Path]) -> Scenario:
    path = Path(json_path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    nodes: Dict[str, ScenarioNode] = {}
    child_refs = set()
    for item in data:
        node = ScenarioNode(
            id=item["id"],
            intents=item.get("intent", []),
            slots=item.get("slot", []),
            actions=item.get("action", []),
            response=item.get("response", ""),
            child_ids=item.get("childnode", []),
        )
        nodes[node.id] = node
        child_refs.update(node.child_ids)
    entry_candidates = [nid for nid in nodes if nid not in child_refs]
    entry_id = entry_candidates[0] if entry_candidates else data[0]["id"]
    return Scenario(name=path.stem, path=path, nodes=nodes, entry_id=entry_id)


def load_scenarios(scenario_dir: Union[str, Path]) -> List[Scenario]:
    directory = Path(scenario_dir)
    paths = sorted(directory.glob("scenario-*.json"))
    return [load_scenario(p) for p in paths]


def normalize_slot_name(slot_token: str) -> str:
    """#服装类型# -> 服装类型"""
    return re.sub(r"^#|#$", "", slot_token.strip())
