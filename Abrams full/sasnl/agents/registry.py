from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentSpec:
    name: str
    tier: str
    transcript_level: str
    domain: str
    gate: str


# Canonical registry derived from .kiro spec so implementation can be expanded safely.
AGENT_REGISTRY: list[AgentSpec] = [
    AgentSpec("FillerWordAgent", "T1_Core", "utterance", "fluency", "always"),
    AgentSpec("RepetitionAgent", "T1_Core", "utterance", "fluency", "always"),
    AgentSpec("SentenceComplexityAgent", "T1_Core", "utterance", "lexical_syntax", "always"),
    AgentSpec("VocabularyDiversityAgent", "T1_Core", "full_transcript", "lexical_syntax", "always"),
    AgentSpec("AgreementErrorAgent", "T1_Core", "utterance", "lexical_syntax", "always"),
    AgentSpec("RunOnSentenceAgent", "T1_Core", "utterance", "lexical_syntax", "always"),
    AgentSpec("SpeechRateRhythmAgent", "T1_Core", "utterance", "prosody", "always"),
    AgentSpec("MLUAgent", "T1_Core", "utterance", "lexical_syntax", "always"),
    AgentSpec("NDWAgent", "T1_Core", "utterance", "lexical_syntax", "always"),
    AgentSpec("LexicalDensityAgent", "T1_Core", "utterance", "lexical_syntax", "always"),
    AgentSpec("PronounReversalAgent", "T1_Core", "utterance", "lexical_syntax", "always"),
    AgentSpec("SemanticCoherenceAgent", "T1_Core", "utterance", "semantics", "always"),
    AgentSpec("ISLAgent", "T1_Core", "utterance", "social_cognition", "always"),
    AgentSpec("ContingentResponseAgent", "T1_Core", "turn_pair", "social_cognition", "always"),
    AgentSpec("FalseStartSelfCorrectionAgent", "T2_Extended", "utterance", "fluency", "repair_signal"),
    AgentSpec("WordFindingPauseAgent", "T2_Extended", "utterance", "fluency", "filled_pause_signal"),
    AgentSpec("CircumlocutionAgent", "T2_Extended", "utterance", "lexical_syntax", "long_noun_chunk"),
    AgentSpec("IncompleteThoughtAgent", "T2_Extended", "utterance", "lexical_syntax", "fragment_signal"),
    AgentSpec("SemanticRelationshipAgent", "T2_Extended", "utterance", "semantics", "logic_cue_word"),
    AgentSpec("FigurativeLanguageAgent", "T2_Extended", "utterance", "semantics", "idiom_signal"),
    AgentSpec("TurnTakingAgent", "T2_Extended", "full_transcript", "pragmatics", "always"),
    AgentSpec("InitiationAgent", "T2_Extended", "turn_pair", "pragmatics", "always"),
    AgentSpec("QuestionUseAgent", "T2_Extended", "utterance", "pragmatics", "question_signal"),
    AgentSpec("WorkingMemoryAgent", "T2_Extended", "topic", "discourse", "always"),
    AgentSpec("InformativenessAgent", "T2_Extended", "utterance", "pragmatics", "always"),
    AgentSpec("PolitenessHedgingAgent", "T2_Extended", "utterance", "pragmatics", "always"),
    AgentSpec("NarrativeMacrostructureAgent", "T2_Extended", "topic", "discourse", "narrative_segment"),
    AgentSpec("CohesionDeviceAgent", "T2_Extended", "utterance", "discourse", "always"),
    AgentSpec("CausalReasoningAgent", "T2_Extended", "utterance", "discourse", "causal_cue"),
    AgentSpec("OrganisationScoreAgent", "T2_Extended", "utterance", "discourse", "always"),
    AgentSpec("PerspectiveTakingAgent", "T2_Extended", "utterance", "social_cognition", "mental_state_verb"),
    AgentSpec("ReferenceTrackingAgent", "T2_Extended", "topic", "social_cognition", "always"),
    AgentSpec("IdiosyncraticWordAgent", "T2_Extended", "utterance", "lexical_syntax", "always"),
    AgentSpec("UtteranceIntonationAgent", "T2_Extended", "utterance", "prosody", "always"),
    AgentSpec("SarcasmDetectionAgent", "Mistral/Claude", "turn_pair", "pragmatics", "mismatch_score > 0.7"),
    AgentSpec("EmpathyAgent", "Mistral/Claude", "turn_pair", "pragmatics", "mismatch_score > 0.7"),
    AgentSpec("GreetingAgent", "Mistral/Claude", "topic", "pragmatics", "greeting_detected"),
    AgentSpec("DepartureAgent", "Mistral/Claude", "topic", "pragmatics", "departure_detected"),
    AgentSpec("TopicManagementAgent", "Mistral/Claude", "topic", "discourse", "topic_shift_signal"),
    AgentSpec("ExecutiveFunctionAgent", "Mistral/Claude", "topic", "discourse", "always"),
    AgentSpec("SemanticTangentialityAgent", "T3_Research", "utterance", "semantics", "optional_enable"),
    AgentSpec("LCMAbstractionAgent", "T3_Research", "utterance", "social_cognition", "optional_enable"),
]
