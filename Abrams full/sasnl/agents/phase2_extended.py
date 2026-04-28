from __future__ import annotations

import statistics
from dataclasses import dataclass

from sasnl.agents.base import Agent
from sasnl.agents.gates import (
    has_filled_pause,
    has_fragment_signal,
    has_idiom_signal,
    has_logic_cue,
    has_long_noun_chunk_proxy,
    has_mental_state_verb,
    has_narrative_segment,
    has_question_signal,
    has_repair_signal,
)
from sasnl.agents.llm_agents import ClaudeStructuredAgent
from sasnl.models import AgentOutput, Utterance


@dataclass
class Phase2AgentSpec:
    name: str
    level: str
    gate_name: str


def gate_passes(gate_name: str, utterances: list[Utterance], all_utterances: list[Utterance]) -> bool:
    if gate_name == "always":
        return True
    if gate_name == "repair_signal":
        return any(has_repair_signal(u.text) for u in utterances)
    if gate_name == "filled_pause_signal":
        return any(has_filled_pause(u.text) for u in utterances)
    if gate_name == "long_noun_chunk":
        return any(has_long_noun_chunk_proxy(u.text) for u in utterances)
    if gate_name == "fragment_signal":
        return any(has_fragment_signal(u.text) for u in utterances)
    if gate_name == "logic_cue_word" or gate_name == "causal_cue":
        return any(has_logic_cue(u.text) for u in utterances)
    if gate_name == "idiom_signal":
        return any(has_idiom_signal(u.text) for u in utterances)
    if gate_name == "question_signal":
        return any(has_question_signal(u.text) for u in utterances)
    if gate_name == "mental_state_verb":
        return any(has_mental_state_verb(u.text) for u in utterances)
    if gate_name == "narrative_segment":
        return has_narrative_segment(all_utterances)
    if gate_name in {"topic_shift_signal", "greeting_detected", "departure_detected"}:
        return True
    return False


class SimpleMetricAgent(Agent):
    def __init__(self, name: str, level: str, scope: str = "student"):
        self.name = name
        self.level = level
        self.scope = scope

    def run(self, context: dict) -> AgentOutput:
        utterances: list[Utterance] = context.get("student_utterances", [])
        wc = sum(u.word_count for u in utterances)
        qn = sum(1 for u in utterances if "?" in u.text)
        avg_len = (wc / len(utterances)) if utterances else 0.0
        return self._output(
            {
                "utterance_count": len(utterances),
                "word_count": wc,
                "question_count": qn,
                "avg_utterance_length_words": round(avg_len, 2),
            }
        )


class TurnTakingAgent(Agent):
    name = "TurnTakingAgent"
    level = "full_transcript"

    def run(self, context: dict) -> AgentOutput:
        all_utts: list[Utterance] = context.get("all_utterances", [])
        student = [u for u in all_utts if u.speaker_role == "student"]
        interviewer = [u for u in all_utts if u.speaker_role == "interviewer"]
        student_words = sum(u.word_count for u in student)
        interviewer_words = sum(u.word_count for u in interviewer)
        ratio = student_words / max(1, interviewer_words)
        per_turn = [u.word_count for u in student]
        balance = statistics.pstdev(per_turn) / max(1e-6, statistics.fmean(per_turn)) if len(per_turn) > 1 else 0.0
        return self._output(
            {
                "student_turns": len(student),
                "interviewer_turns": len(interviewer),
                "student_words": student_words,
                "interviewer_words": interviewer_words,
                "word_balance_ratio": round(ratio, 3),
                "turn_balance_cv": round(balance, 3),
                "clinical_flag": ratio < 0.3 or ratio > 0.7,
            }
        )


def build_phase2_agent_specs() -> list[Phase2AgentSpec]:
    return [
        Phase2AgentSpec("FalseStartSelfCorrectionAgent", "utterance", "repair_signal"),
        Phase2AgentSpec("WordFindingPauseAgent", "utterance", "filled_pause_signal"),
        Phase2AgentSpec("CircumlocutionAgent", "utterance", "long_noun_chunk"),
        Phase2AgentSpec("IncompleteThoughtAgent", "utterance", "fragment_signal"),
        Phase2AgentSpec("SemanticRelationshipAgent", "utterance", "logic_cue_word"),
        Phase2AgentSpec("FigurativeLanguageAgent", "utterance", "idiom_signal"),
        Phase2AgentSpec("InitiationAgent", "turn_pair", "always"),
        Phase2AgentSpec("QuestionUseAgent", "utterance", "question_signal"),
        Phase2AgentSpec("WorkingMemoryAgent", "topic", "always"),
        Phase2AgentSpec("InformativenessAgent", "utterance", "always"),
        Phase2AgentSpec("PolitenessHedgingAgent", "utterance", "always"),
        Phase2AgentSpec("NarrativeMacrostructureAgent", "topic", "narrative_segment"),
        Phase2AgentSpec("CohesionDeviceAgent", "utterance", "always"),
        Phase2AgentSpec("CausalReasoningAgent", "utterance", "causal_cue"),
        Phase2AgentSpec("OrganisationScoreAgent", "utterance", "always"),
        Phase2AgentSpec("PerspectiveTakingAgent", "utterance", "mental_state_verb"),
        Phase2AgentSpec("ReferenceTrackingAgent", "topic", "always"),
        Phase2AgentSpec("IdiosyncraticWordAgent", "utterance", "always"),
        Phase2AgentSpec("UtteranceIntonationAgent", "utterance", "always"),
    ]


def run_phase2_agents(context: dict, llm_client) -> dict[str, AgentOutput]:
    outputs: dict[str, AgentOutput] = {}
    student_utterances: list[Utterance] = context.get("student_utterances", [])
    all_utterances: list[Utterance] = context.get("all_utterances", [])

    outputs["TurnTakingAgent"] = TurnTakingAgent().run(context)

    for spec in build_phase2_agent_specs():
        if not gate_passes(spec.gate_name, student_utterances, all_utterances):
            outputs[spec.name] = AgentOutput.skipped(spec.name, spec.level, "student", f"gate_not_met:{spec.gate_name}")
            continue

        base_metrics = SimpleMetricAgent(spec.name, spec.level).run(context)
        llm_agent = ClaudeStructuredAgent(llm_client, level=spec.level, name=spec.name)
        llm_out = llm_agent.run(context)

        merged = AgentOutput(
            agent_name=spec.name,
            version="1.0",
            transcript_level=spec.level,
            speaker_scope="student",
            status="completed",
            computed_at=llm_out.computed_at,
            metrics={**base_metrics.metrics, **llm_out.metrics, "gate": spec.gate_name},
            interpretation=llm_out.interpretation,
            evidence=base_metrics.evidence + llm_out.evidence,
        )
        outputs[spec.name] = merged

    return outputs
