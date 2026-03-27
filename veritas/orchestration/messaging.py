"""Message passing abstraction for agent communication."""
from __future__ import annotations
from dataclasses import dataclass
from veritas.core.result import AgentFinding

@dataclass
class AgentMessage:
    agent_name: str
    finding: AgentFinding

class MessageBus:
    def __init__(self):
        self._messages: list[AgentMessage] = []

    def send(self, agent_name: str, finding: AgentFinding) -> None:
        self._messages.append(AgentMessage(agent_name=agent_name, finding=finding))

    def collect(self) -> list[AgentFinding]:
        return [m.finding for m in self._messages]

    def clear(self) -> None:
        self._messages.clear()
