"""PM：对话策略，决定下一动作（追问槽位 / 回复 / 切换节点）。"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from dst import DST, DialogPhase
from nlu import NLU, NLUResult
from scenario_loader import Scenario, ScenarioNode, SlotConfig


class PolicyActionType(str, Enum):
    GREET = "greet"
    ASK_SLOT = "ask_slot"
    RESPOND = "respond"
    CLARIFY = "clarify"
    FINISHED = "finished"


@dataclass
class PolicyAction:
    type: PolicyActionType
    response_template: str = ""
    slot_query: str = ""
    asking_slot: Optional[str] = None
    actions: List[str] = field(default_factory=list)
    node: Optional[ScenarioNode] = None


class PolicyManager:
    def __init__(
        self,
        scenarios: List[Scenario],
        slot_configs: dict[str, SlotConfig],
        nlu: NLU,
        dst: DST,
    ):
        self.scenarios = scenarios
        self.slot_configs = slot_configs
        self.nlu = nlu
        self.dst = dst

    def _entry_candidates(self) -> list[tuple[Scenario, ScenarioNode]]:
        return [(s, s.nodes[s.entry_id]) for s in self.scenarios]

    def _slot_query(self, slot_token: str) -> str:
        cfg = self.slot_configs.get(slot_token)
        if cfg:
            return cfg.query
        return f"请提供{slot_token}"

    def _continue_slot_or_respond(self, scenario: Scenario, node: ScenarioNode) -> PolicyAction:
        """在当前节点上根据 DST 已填槽位决定追问或输出业务回复。"""
        self.dst.set_pending_from_node(node)
        if self.dst.is_collecting():
            first = self.dst.state.pending_slots[0]
            return PolicyAction(
                type=PolicyActionType.ASK_SLOT,
                slot_query=self._slot_query(first),
                asking_slot=self.dst.state.asking_slot,
            )
        if not node.child_ids:
            self.dst.mark_finished()
            return PolicyAction(
                type=PolicyActionType.FINISHED,
                response_template=node.response,
                actions=list(node.actions),
                node=node,
            )
        self.dst.state.phase = DialogPhase.WAITING_NEXT_INTENT
        return PolicyAction(
            type=PolicyActionType.RESPOND,
            response_template=node.response,
            actions=list(node.actions),
            node=node,
        )

    def _activate_node(self, scenario: Scenario, node: ScenarioNode) -> PolicyAction:
        self.dst.state.last_actions = list(node.actions)
        self.dst.state.scenario = scenario
        self.dst.state.current_node = node
        return self._continue_slot_or_respond(scenario, node)

    def decide(self, user_text: str) -> PolicyAction:
        state = self.dst.state
        text = user_text.strip()

        if state.phase == DialogPhase.FINISHED:
            return PolicyAction(
                type=PolicyActionType.FINISHED,
                response_template="本次流程已结束，如需重新开始请输入新的业务意图。",
            )

        if state.phase == DialogPhase.COLLECTING_SLOTS and state.pending_slots:
            extracted = self.nlu.extract_slots(text, state.pending_slots, state.slots)
            if extracted:
                for name, value in extracted.items():
                    self.dst.fill_single_slot(name, value)
            else:
                slot_token = state.pending_slots[0]
                slot_name = self.dst._token_to_name(slot_token)
                cfg = self.slot_configs.get(slot_token)
                hint = ""
                if cfg and not cfg.is_regex and cfg.values:
                    hint = f"（例如：{'、'.join(cfg.values[:6])}）"
                elif cfg and cfg.is_regex:
                    hint = "（例如：8点、19点）"
                return PolicyAction(
                    type=PolicyActionType.CLARIFY,
                    slot_query=self._slot_query(slot_token) + "（未能识别，请换个说法）" + hint,
                    asking_slot=slot_name,
                )
            node = state.current_node
            scenario = state.scenario
            if node and scenario:
                return self._continue_slot_or_respond(scenario, node)
            return PolicyAction(type=PolicyActionType.CLARIFY, slot_query="状态异常，请重新开始。")

        candidates: list[tuple[Scenario, ScenarioNode]] = []
        if state.current_node and state.scenario and state.phase == DialogPhase.WAITING_NEXT_INTENT:
            for child in self.dst.child_candidates():
                candidates.append((state.scenario, child))
        else:
            candidates = self._entry_candidates()

        nlu_result: NLUResult = self.nlu.parse(
            text,
            candidates,
            existing_slots=state.slots,
        )

        if nlu_result.matched_node and nlu_result.matched_scenario:
            if nlu_result.slots:
                self.dst.update_slots(nlu_result.slots)
            return self._activate_node(nlu_result.matched_scenario, nlu_result.matched_node)

        if state.phase == DialogPhase.WAITING_NEXT_INTENT:
            hints = []
            for child in self.dst.child_candidates():
                hints.extend(child.intents)
            hint = " / ".join(hints) if hints else "相关业务意图"
            return PolicyAction(
                type=PolicyActionType.CLARIFY,
                slot_query=f"暂未理解，您可以尝试说：{hint}",
            )

        return PolicyAction(
            type=PolicyActionType.GREET,
            response_template="您好，我是任务型对话助手。您可以说「我要看电影」或「我要买衣服」开始。",
        )
