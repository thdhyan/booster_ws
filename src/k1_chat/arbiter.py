"""Velocity arbiter: manual control always beats chat. TDD stub: tests/test_arbiter.py defines the contract."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Twist:
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0


class VelocityArbiter:
    def __init__(self, max_vx=0.4, max_vy=0.2, max_vyaw=0.5, chat_timeout=1.0, manual_hold=2.0, clock=None):
        raise NotImplementedError

    def manual(self, t: Twist) -> None: ...
    def chat(self, t: Twist) -> None: ...
    def estop(self) -> None: ...
    def release(self) -> None: ...
    def output(self) -> Twist: ...
