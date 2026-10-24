"""DST：对话状态跟踪。"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from scenario_loader import Scenario, ScenarioNode, normalize_slot_name


class DialogPhase(str, Enum):
    IDLE = "idle"
    COLLECTING_SLOTS = "collecting_slots"
    WAITING_NEXT_INTENT = "waiting_next_intent"
    FINISHED = "finished"


@dataclass
class DialogState:
    scenario: Optional[Scenario] = None
    current_node: Optional[ScenarioNode] = None
    slots: Dict[str, str] = field(default_factory=dict)
    pending_slots: List[str] = field(default_factory=list)
    asking_slot: Optional[str] = None
    phase: DialogPhase = DialogPhase.IDLE
    last_actions: List[str] = field(default_factory=list)


class DST:
    def __init__(self):
        self.state = DialogState()

    def reset(self) -> None:
        self.state = DialogState()

    def _token_to_name(self, slot_token: str) -> str:
        return normalize_slot_name(slot_token)

    def set_pending_from_node(self, node: ScenarioNode) -> None:
        pending = []
        for token in node.slots:
            name = self._token_to_name(token)
            if name not in self.state.slots:
                pending.append(token)
        self.state.pending_slots = pending
        self.state.asking_slot = self._token_to_name(pending[0]) if pending else None
        self.state.phase = (
            DialogPhase.COLLECTING_SLOTS if pending else DialogPhase.WAITING_NEXT_INTENT
        )

    def update_slots(self, new_slots: Dict[str, str]) -> None:
        for k, v in new_slots.items():
            if v:
                self.state.slots[k] = v
        if self.state.current_node:
            self.set_pending_from_node(self.state.current_node)

    def fill_single_slot(self, slot_name: str, value: str) -> None:
        if value:
            self.state.slots[slot_name] = value
        if self.state.current_node:
            self.set_pending_from_node(self.state.current_node)

    def activate_node(self, scenario: Scenario, node: ScenarioNode) -> None:
        self.state.scenario = scenario
        self.state.current_node = node
        self.set_pending_from_node(node)

    def mark_finished(self) -> None:
        self.state.phase = DialogPhase.FINISHED
        self.state.pending_slots = []
        self.state.asking_slot = None

    def child_candidates(self) -> List[ScenarioNode]:
        node = self.state.current_node
        scenario = self.state.scenario
        if not node or not scenario:
            return []
        return [scenario.nodes[cid] for cid in node.child_ids if cid in scenario.nodes]

    def is_collecting(self) -> bool:
        return self.state.phase == DialogPhase.COLLECTING_SLOTS and bool(
            self.state.pending_slots
        )
