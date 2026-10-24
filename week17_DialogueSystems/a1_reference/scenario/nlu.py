"""NLU：基于编辑距离的意图识别与槽位抽取。"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from scenario_loader import Scenario, ScenarioNode, SlotConfig, normalize_slot_name
from utils import best_match_in_candidates, best_substring_match, normalized_similarity, regex_slot_value


@dataclass
class NLUResult:
    intent: Optional[str] = None
    intent_score: float = 0.0
    matched_node: Optional[ScenarioNode] = None
    matched_scenario: Optional[Scenario] = None
    slots: Dict[str, str] = field(default_factory=dict)


class NLU:
    def __init__(
        self,
        scenarios: List[Scenario],
        slot_configs: Dict[str, SlotConfig],
        intent_threshold: float = 0.5,
    ):
        self.scenarios = scenarios
        self.slot_configs = slot_configs
        self.intent_threshold = intent_threshold

    def _slot_key(self, slot_token: str) -> str:
        return normalize_slot_name(slot_token)

    def _slot_token(self, slot_name: str) -> str:
        return f"#{slot_name}#"

    def recognize_intent(
        self,
        text: str,
        candidate_nodes: List[tuple[Scenario, ScenarioNode]],
    ) -> tuple[Optional[str], float, Optional[Scenario], Optional[ScenarioNode]]:
        text = text.strip()
        text_l = text.lower()
        scored: list[tuple[str, float, Scenario, ScenarioNode, bool]] = []
        for scenario, node in candidate_nodes:
            for intent in node.intents:
                intent_l = intent.strip().lower()
                exact = text_l == intent_l
                contains = intent_l in text_l
                score = 1.0 if exact else (0.92 if contains else normalized_similarity(text_l, intent_l))
                scored.append((intent, score, scenario, node, exact))
        if not scored:
            return None, 0.0, None, None
        scored.sort(key=lambda x: (x[4], x[1]), reverse=True)
        best = scored[0]
        if best[1] < self.intent_threshold:
            return None, 0.0, None, None
        if len(scored) > 1 and not best[4]:
            second = scored[1]
            if second[1] >= self.intent_threshold and (best[1] - second[1]) < 0.08:
                return None, 0.0, None, None
        return best[0], best[1], best[2], best[3]

    def extract_slots(
        self,
        text: str,
        slot_tokens: List[str],
        existing: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        existing = existing or {}
        found: Dict[str, str] = {}
        for token in slot_tokens:
            name = self._slot_key(token)
            if name in existing:
                continue
            cfg = self.slot_configs.get(self._slot_token(name)) or self.slot_configs.get(token)
            if not cfg:
                continue
            if cfg.is_regex and cfg.regex_pattern:
                val = regex_slot_value(text, cfg.regex_pattern)
                if val:
                    found[name] = val
            else:
                val = best_match_in_candidates(text, cfg.values, threshold=0.5)
                if not val:
                    val = best_substring_match(text, cfg.values)
                if val:
                    found[name] = val
        return found

    def parse(
        self,
        text: str,
        candidate_nodes: List[tuple[Scenario, ScenarioNode]],
        target_slot_tokens: Optional[List[str]] = None,
        existing_slots: Optional[Dict[str, str]] = None,
    ) -> NLUResult:
        intent, score, scenario, node = self.recognize_intent(text, candidate_nodes)
        slot_tokens = target_slot_tokens or (node.slots if node else [])
        slots = self.extract_slots(text, slot_tokens, existing_slots)
        return NLUResult(
            intent=intent,
            intent_score=score,
            matched_node=node,
            matched_scenario=scenario,
            slots=slots,
        )

    def parse_slot_answer(
        self,
        text: str,
        slot_token: str,
        existing_slots: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        return self.extract_slots(text, [slot_token], existing_slots)
