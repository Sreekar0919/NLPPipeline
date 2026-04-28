from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from sasnl.agents.full_battery import FullBatteryRunner
from sasnl.agents.llm_agents import NarratorAgent
from sasnl.asr import parse_transcript, transcribe_audio
from sasnl.config import PipelineConfig
from sasnl.domain_aggregator import aggregate_domains
from sasnl.feature_extractor import extract_features
from sasnl.llm import BedrockClaudeClient
from sasnl.output_writer import write_session_output
from sasnl.prosody import compute_session_baseline, interpret_prosody
from sasnl.segmenter import segment_words
from sasnl.temporal_summary import build_temporal_summary
from sasnl.topic_segmenter import build_topic_segments


def _serialize_agents(agents: dict) -> dict:
    out = {}
    for k, v in agents.items():
        out[k] = asdict(v)
    return out


class PipelineOrchestrator:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.llm = BedrockClaudeClient(self.config.model, strict_mode=self.config.strict_mode)

    def _detect_input_type(self, input_source: str) -> str:
        """Auto-detect whether input is audio or transcript."""
        path = Path(input_source)
        
        # Check if it's a valid audio file
        audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
        if path.exists() and path.suffix.lower() in audio_extensions:
            return "audio"
        
        # Check if it's a JSON file
        if path.exists() and path.suffix.lower() == '.json':
            return "transcript"
        
        # Check if it looks like JSON
        if input_source.strip().startswith(('[', '{')):
            return "transcript"
        
        # Otherwise treat as text or file path
        return "transcript"

    def run(
        self,
        input_source: str | None = None,
        student_id: str | None = None,
        session_type: str | None = None,
        input_type: str = "auto",
        # Legacy parameters for backward compatibility
        audio_file: str | None = None,
    ) -> Path:
        # Support both old signature (audio_file) and new signature (input_source)
        if input_source is None and audio_file is not None:
            input_source = audio_file
            input_type = "audio"
        
        if input_source is None or student_id is None:
            raise ValueError("input_source and student_id are required")
        
        session_type = session_type or self.config.assessment_type_default
        
        # Determine input type
        if input_type == "auto":
            input_type = self._detect_input_type(input_source)
        
        # Get words based on input type
        if input_type == "audio":
            words = transcribe_audio(Path(input_source), strict_mode=self.config.strict_mode)
        elif input_type == "transcript":
            words = parse_transcript(input_source)
        else:
            raise ValueError(f"Unknown input_type: {input_type}. Must be 'audio', 'transcript', or 'auto'.")
        
        session_duration_ms = max((w.end_ms for w in words), default=1)
        utterances = segment_words(words, session_duration_ms=session_duration_ms, gap_ms=self.config.speaker_gap_ms)

        feature_store = extract_features(utterances, strict_mode=self.config.strict_mode)
        baseline = compute_session_baseline(utterances)
        interpret_prosody(utterances, baseline, pause_flag_ms=self.config.pause_flag_ms)

        student_utts = [u for u in utterances if u.speaker_role == "student"]
        context = {
            "student_utterances": student_utts,
            "all_utterances": utterances,
            "assessment_type": session_type,
        }

        battery = FullBatteryRunner(
            llm_client=self.llm,
            mismatch_threshold=self.config.mismatch_gate_threshold,
            enable_t3=self.config.enable_t3_research_agents,
        )
        agents = battery.run(context)

        topic_segments = build_topic_segments(utterances, self.config.model.analyst_model_id)
        temporal_summary = build_temporal_summary(utterances, agents)
        domains, overall_profile = aggregate_domains(agents)
        overall_profile["report_generated_at"] = datetime.now(timezone.utc).isoformat()

        narrator = NarratorAgent(self.llm)
        flagged = [
            a
            for a in agents.values()
            if a.status == "completed" and a.interpretation and a.interpretation.get("confidence", 0) >= 0.5
        ]
        narrative = narrator.generate(
            {
                "session": {"student_id": student_id, "session_type": session_type},
                "temporal_summary": temporal_summary,
                "flagged_agent_outputs": [
                    {
                        "agent_name": f.agent_name,
                        "severity": f.interpretation.get("severity", "mild"),
                        "clinical_note": f.interpretation.get("clinical_note", ""),
                        "confidence": f.interpretation.get("confidence", 0.5),
                    }
                    for f in flagged
                ],
            }
        )

        transcript_block = {"utterances": [u.to_transcript_dict() for u in utterances]}
        transcript_word_count = sum(u.word_count for u in utterances)
        
        # Dynamically determine speaker IDs based on actual utterances
        speaker_id_map = {}
        for u in utterances:
            if u.speaker_role not in speaker_id_map:
                speaker_id_map[u.speaker_role] = u.speaker_id
        
        data_quality = {
            "valid_input": True,
            "speaker_diarization_confidence": 0.8,
            "transcript_word_count": transcript_word_count,
            "transcript_utterance_count": len(utterances),
            "student_utterance_count": sum(1 for u in utterances if u.speaker_role == "student"),
            "interviewer_utterance_count": sum(1 for u in utterances if u.speaker_role == "interviewer"),
            "student_word_count": sum(u.word_count for u in utterances if u.speaker_role == "student"),
            "interviewer_word_count": sum(u.word_count for u in utterances if u.speaker_role == "interviewer"),
            "speaker_ids": speaker_id_map,
            "issues": [],
            "flags": [],
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }

        payload = {
            "schema_version": "1.1",
            "session": {
                "session_id": f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                "student_id": student_id,
                "student_age_months": None,
                "session_type": session_type,
                "baseline_session_id": None,
                "session_duration_s": round(session_duration_ms / 1000.0, 2),
            },
            "data_quality": data_quality,
            "transcript": transcript_block,
            "feature_store": feature_store,
            "topic_segments": topic_segments,
            "agents": _serialize_agents(agents),
            "domains": domains,
            "overall_profile": overall_profile,
            "temporal_summary": temporal_summary,
            "narrative": narrative,
        }
        return write_session_output(self.config.output_dir, payload)
