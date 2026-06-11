"""Small trajectory containers for grouped GRPO rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Step:
    obs: dict
    action: int
    logprob: float
    reward: float
    status: str
    info: dict


@dataclass
class Trajectory:
    steps: list[Step] = field(default_factory=list)
    done: bool = False
    died: bool = False

    @property
    def total_reward(self) -> float:
        return float(sum(step.reward for step in self.steps))

    @property
    def actions(self) -> list[int]:
        return [step.action for step in self.steps]

    def append(self, obs: dict, action: int, logprob: float, reward: float, status: str, info: dict) -> None:
        self.steps.append(Step(obs, action, logprob, reward, status, info))
        self.done = status != "RUNNING"
        self.died = status == "DEAD"
