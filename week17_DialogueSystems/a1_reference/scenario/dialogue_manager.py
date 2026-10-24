"""对话管理器：串联 NLU -> DST -> PM -> NLG。"""

from pathlib import Path

from dst import DST
from nlg import NLG
from nlu import NLU
from pm import PolicyActionType, PolicyManager
from scenario_loader import load_scenarios, load_slot_config


class DialogueManager:
    def __init__(self, scenario_dir: str | Path | None = None):
        base = Path(scenario_dir or Path(__file__).resolve().parent)
        self.scenarios = load_scenarios(base)
        excel_path = base / "slot_fitting_templet.xlsx"
        self.slot_configs = load_slot_config(excel_path)
        self.dst = DST()
        self.nlu = NLU(self.scenarios, self.slot_configs)
        self.pm = PolicyManager(self.scenarios, self.slot_configs, self.nlu, self.dst)
        self.nlg = NLG()

    def reset(self) -> None:
        self.dst.reset()

    def turn(self, user_text: str) -> str:
        policy = self.pm.decide(user_text)
        reply = self.nlg.generate(policy, self.dst.state.slots)
        if policy.actions:
            action_msg = "【系统动作】" + "；".join(policy.actions)
            reply = f"{reply}\n{action_msg}"
        return reply
