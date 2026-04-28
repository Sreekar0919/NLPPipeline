from __future__ import annotations

import json
from pathlib import Path

import pytest

from sasnl.config import ModelConfig, PipelineConfig
from sasnl.llm import BedrockClaudeClient
from sasnl.pipeline import PipelineOrchestrator


def test_no_strict_pipeline_generates_full_agent_battery() -> None:
    root = Path(__file__).resolve().parents[1]
    audio = root / "combined_audio.wav"
    cfg = PipelineConfig(strict_mode=False, enable_t3_research_agents=False)
    orchestrator = PipelineOrchestrator(cfg)
    output_path = orchestrator.run(input_source=str(audio), student_id="student_test", session_type="SCS")
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1.1"
    assert "agents" in payload and isinstance(payload["agents"], dict)
    assert "domains" in payload and isinstance(payload["domains"], dict)

    required_agents = [
        "FillerWordAgent",
        "FalseStartSelfCorrectionAgent",
        "RepetitionAgent",
        "DelayedRepetitionAgent",
        "SpeechRateRhythmAgent",
        "MLUAgent",
        "VocabularyDiversityAgent",
        "NDWAgent",
        "LexicalDensityAgent",
        "SentenceComplexityAgent",
        "AgreementErrorAgent",
        "PronounReversalAgent",
        "IdiosyncraticWordAgent",
        "SemanticCoherenceAgent",
        "GlobalCoherenceAgent",
        "SentimentISLAgent",
        "SemanticRelationshipAgent",
        "FigurativeLanguageAgent",
        "GreetingAgent",
        "DepartureAgent",
        "InitiationAgent",
        "TurnTakingAgent",
        "QuestionUseAgent",
        "InformativenessAgent",
        "PolitenessHedgingAgent",
        "TopicManagementAgent",
        "WorkingMemoryAgent",
        "NarrativeMacrostructureAgent",
        "CohesionDeviceAgent",
        "CausalReasoningAgent",
        "OrganisationScoreAgent",
        "ISLAgent",
        "PerspectiveTakingAgent",
        "ReferenceTrackingAgent",
        "ContingentResponseAgent",
        "UtteranceIntonationAgent",
        "SarcasmDetectionAgent",
        "EmpathyAgent",
        "ExecutiveFunctionAgent",
    ]
    for agent_name in required_agents:
        assert agent_name in payload["agents"], f"missing agent output: {agent_name}"


def test_strict_mode_bedrock_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummySession:
        def get_credentials(self):
            return None

    import boto3

    monkeypatch.setattr(boto3.session, "Session", lambda: DummySession())
    with pytest.raises(RuntimeError, match="strict_mode requires valid AWS credentials"):
        BedrockClaudeClient(ModelConfig(), strict_mode=True)


def test_pipeline_works_with_custom_transcript_format() -> None:
    """Test that pipeline can process transcripts in custom format with speaker and word timings."""
    cfg = PipelineConfig(strict_mode=False, enable_t3_research_agents=False)
    orchestrator = PipelineOrchestrator(cfg)
    
    # Custom transcript format
    transcript = """[SPEAKER 1] GO DO YOU HEAR [00:00:00,000 --> 00:00:01,700]
timestamp: GO(0.000-0.680), DO(1.100-1.380), YOU(1.380-1.500), HEAR(1.500-1.700)
[SPEAKER 2] YES I HEAR YOU [00:00:02,000 --> 00:00:03,500]
timestamp: YES(2.000-2.300), I(2.300-2.500), HEAR(2.500-2.800), YOU(2.800-3.500)"""
    
    output_path = orchestrator.run(
        input_source=transcript,
        student_id="student_test",
        session_type="SCS",
        input_type="transcript"
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    
    assert payload["schema_version"] == "1.1"
    assert "agents" in payload and isinstance(payload["agents"], dict)
    assert len(payload["transcript"]["utterances"]) > 0


def test_pipeline_works_with_json_transcript() -> None:
    """Test that pipeline can process JSON transcripts."""
    cfg = PipelineConfig(strict_mode=False, enable_t3_research_agents=False)
    orchestrator = PipelineOrchestrator(cfg)
    
    # JSON transcript format
    transcript = '[{"word": "hello", "start_ms": 0, "end_ms": 250, "speaker_id": "SPK_A"}, {"word": "there", "start_ms": 300, "end_ms": 550, "speaker_id": "SPK_A"}]'
    
    output_path = orchestrator.run(
        input_source=transcript,
        student_id="student_test",
        session_type="SCS",
        input_type="transcript"
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    
    assert payload["schema_version"] == "1.1"
    assert "agents" in payload
    assert len(payload["transcript"]["utterances"]) > 0


def test_pipeline_auto_detects_transcript() -> None:
    """Test that pipeline auto-detects JSON transcripts."""
    cfg = PipelineConfig(strict_mode=False, enable_t3_research_agents=False)
    orchestrator = PipelineOrchestrator(cfg)
    
    transcript = '[{"word": "hi", "start_ms": 0, "end_ms": 300, "speaker_id": "SPK_A"}]'
    
    output_path = orchestrator.run(
        input_source=transcript,
        student_id="student_test",
        session_type="SCS",
        input_type="auto"
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    
    assert payload["schema_version"] == "1.1"
    assert "agents" in payload
