from __future__ import annotations

import json

from sasnl.agents.base import Agent
from sasnl.llm import BedrockClaudeClient
from sasnl.models import AgentOutput, Utterance


class ClaudeStructuredAgent(Agent):
    prompt_label: str = ""

    def __init__(self, llm: BedrockClaudeClient, level: str, name: str):
        self.llm = llm
        self.level = level
        self.name = name

    def run(self, context: dict) -> AgentOutput:
        utterances: list[Utterance] = context.get("student_utterances", [])
        payload = {
            "prompt_label": self.prompt_label,
            "utterances": [
                {
                    "utterance_id": u.utterance_id,
                    "text": u.text,
                    "prosody_text": u.prosody_text,
                    "timestamp": [u.start_ms, u.end_ms],
                }
                for u in utterances
            ],
            "instructions": "Return strict JSON with keys: functional_label, severity, clinical_note, confidence.",
        }
        raw = self.llm.invoke_json(
            prompt=json.dumps(payload),
            system_prompt="You are a clinical language analysis assistant. Return only JSON.",
        )
        interpretation = {
            "source": "llm",
            "model": self.llm.config.analyst_model_id,
            "functional_label": raw.get("functional_label", "unknown"),
            "severity": raw.get("severity", "mild"),
            "severity_scale": "none | mild | moderate | severe",
            "clinical_note": raw.get("clinical_note", "")[:500],
            "confidence": float(raw.get("confidence", 0.5)),
        }
        return self._output(metrics={"call": "bedrock_claude"}, interpretation=interpretation)


class NarratorAgent(ClaudeStructuredAgent):
    def __init__(self, llm: BedrockClaudeClient):
        super().__init__(llm=llm, level="full_transcript", name="Narrator")

    def generate(self, narrative_input: dict) -> str:
        result = self.llm.invoke_json(
            prompt=json.dumps(narrative_input),
            system_prompt=(
                "Write four short clinical paragraphs: fluency, language form, pragmatics, prosody-pragmatics. "
                "Third person, past tense, no diagnosis. Return JSON: {narrative: string}."
            ),
            narrator=True,
        )
        return result.get("narrative", result.get("raw_text", "Narrative unavailable."))
