from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone

from sasnl.models import AgentOutput


class Agent(ABC):
    name: str = "Agent"
    level: str = "utterance"
    scope: str = "student"

    @abstractmethod
    def run(self, context: dict) -> AgentOutput:
        raise NotImplementedError

    def _output(self, metrics: dict, interpretation: dict | None = None, evidence: list | None = None) -> AgentOutput:
        return AgentOutput(
            agent_name=self.name,
            version="1.0",
            transcript_level=self.level,
            speaker_scope=self.scope,
            status="completed",
            computed_at=datetime.now(timezone.utc).isoformat(),
            metrics={"source": "nlp", **metrics},
            interpretation=interpretation or {},
            evidence=evidence or [],
        )
