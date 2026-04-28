from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Literal

Severity = Literal["none", "mild", "moderate", "severe"]
TranscriptLevel = Literal["utterance", "turn_pair", "topic", "full_transcript"]
StatusType = Literal["completed", "skipped", "error"]


@dataclass
class WordTiming:
    word: str
    start_ms: int
    end_ms: int
    speaker_id: str = "unknown"
    speaker_role: str | None = None  # Can be "student", "interviewer", "clinician", or None if not specified


@dataclass
class Token:
    index: int
    text: str
    lemma: str = ""
    pos: str = ""
    dep: str = ""


@dataclass
class EvidenceItem:
    evidence_type: str
    utterance_id: str
    utterance_text: str
    start_ms: int
    end_ms: int
    char_start: int = 0
    char_end: int = 0
    surface: str = ""
    feature_type: str = ""
    position_in_utterance: str = ""
    llm_note: str = ""


@dataclass
class AgentOutput:
    agent_name: str
    version: str
    transcript_level: TranscriptLevel
    speaker_scope: str
    status: StatusType
    computed_at: str
    metrics: dict[str, Any]
    interpretation: dict[str, Any] = field(default_factory=dict)
    evidence: list[EvidenceItem] = field(default_factory=list)

    @staticmethod
    def skipped(name: str, level: TranscriptLevel, scope: str, reason: str) -> "AgentOutput":
        return AgentOutput(
            agent_name=name,
            version="1.0",
            transcript_level=level,
            speaker_scope=scope,
            status="skipped",
            computed_at=datetime.now(timezone.utc).isoformat(),
            metrics={"source": "nlp", "skip_reason": reason},
            interpretation={},
            evidence=[],
        )


@dataclass
class Utterance:
    utterance_id: str
    speaker_id: str
    speaker_role: str
    start_ms: int
    end_ms: int
    turn_index: int
    text: str
    words: list[WordTiming]
    t_norm: float = 0.0
    start_time_s: float = 0.0
    end_time_s: float = 0.0
    word_count: int = 0
    tokens: list[Token] = field(default_factory=list)
    audio_features: dict[str, Any] = field(default_factory=dict)
    prosody_text: str = ""
    nlp_features: dict[str, Any] = field(default_factory=dict)
    spacy_doc: Any = None
    text_embedding: list[float] = field(default_factory=list)
    agent_outputs: dict[str, AgentOutput] = field(default_factory=dict)
    clinical_flags: list[dict[str, Any]] = field(default_factory=list)

    def to_transcript_dict(self) -> dict[str, Any]:
        return {
            "utterance_id": self.utterance_id,
            "speaker_id": self.speaker_id,
            "speaker_role": self.speaker_role,
            "text": self.text,
            "word_count": self.word_count,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "tokens": [asdict(t) for t in self.tokens],
        }
