"""NLG：模板填槽生成自然语言回复。"""

import re

from pm import PolicyAction, PolicyActionType


class NLG:
    def generate(self, policy_action: PolicyAction, slots: dict[str, str]) -> str:
        if policy_action.type == PolicyActionType.ASK_SLOT:
            return policy_action.slot_query
        if policy_action.type == PolicyActionType.CLARIFY:
            return policy_action.slot_query
        if policy_action.type == PolicyActionType.GREET:
            return policy_action.response_template
        if policy_action.type in (PolicyActionType.RESPOND, PolicyActionType.FINISHED):
            return self._fill_template(policy_action.response_template, slots)
        return "好的。"

    def _fill_template(self, template: str, slots: dict[str, str]) -> str:
        if not template:
            return "好的。"
        result = template
        for name, value in slots.items():
            result = result.replace(f"#{name}#", str(value))
        result = re.sub(r"#[^#]+#", "（未填写）", result)
        return result
