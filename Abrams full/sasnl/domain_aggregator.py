from __future__ import annotations

SEVERITY_ORDER = ["none", "mild", "moderate", "severe"]

DOMAIN_MAP = {
    "fluency": [
        "FillerWordAgent",
        "FalseStartSelfCorrectionAgent",
        "RepetitionAgent",
        "DelayedRepetitionAgent",
        "SpeechRateRhythmAgent",
    ],
    "lexical_syntax": [
        "MLUAgent",
        "VocabularyDiversityAgent",
        "NDWAgent",
        "LexicalDensityAgent",
        "SentenceComplexityAgent",
        "AgreementErrorAgent",
        "PronounReversalAgent",
        "IdiosyncraticWordAgent",
    ],
    "semantics": [
        "SemanticCoherenceAgent",
        "GlobalCoherenceAgent",
        "SentimentISLAgent",
        "SemanticRelationshipAgent",
        "FigurativeLanguageAgent",
        "SemanticTangentialityAgent",
    ],
    "pragmatics": [
        "GreetingAgent",
        "DepartureAgent",
        "TurnTakingAgent",
        "InitiationAgent",
        "QuestionUseAgent",
        "InformativenessAgent",
        "PolitenessHedgingAgent",
        "SarcasmDetectionAgent",
        "EmpathyAgent",
    ],
    "discourse": ["WorkingMemoryAgent", "NarrativeMacrostructureAgent", "CohesionDeviceAgent", "CausalReasoningAgent", "OrganisationScoreAgent", "TopicManagementAgent", "ExecutiveFunctionAgent"],
    "social_cognition": ["ISLAgent", "PerspectiveTakingAgent", "ReferenceTrackingAgent", "ContingentResponseAgent", "LCMAbstractionAgent"],
    "prosody": ["SpeechRateRhythmAgent", "UtteranceIntonationAgent"],
}


def _sev_idx(sev: str) -> int:
    try:
        return SEVERITY_ORDER.index(sev)
    except ValueError:
        return 0


def aggregate_domains(agent_outputs: dict) -> tuple[dict, dict]:
    domains = {}
    severity_by_domain = {}

    for domain, names in DOMAIN_MAP.items():
        contributing = [agent_outputs[n] for n in names if n in agent_outputs]
        completed = [a for a in contributing if a.status == "completed"]
        severities = [a.interpretation.get("severity", "none") for a in completed if a.interpretation]
        max_sev = max(severities, key=_sev_idx) if severities else "none"
        conf = (sum(float(a.interpretation.get("confidence", 0.5)) for a in completed if a.interpretation) / max(1, len(completed)))
        if len(completed) < 2:
            conf = min(conf, 0.49)
        domains[domain] = {
            "contributing_agents": [a.agent_name for a in contributing],
            "severity": max_sev,
            "severity_scale": "none | mild | moderate | severe",
            "summary": "Auto-aggregated from Phase 2 outputs.",
            "strengths": [],
            "areas_for_growth": [],
            "confidence": round(conf, 3),
        }
        severity_by_domain[domain] = max_sev

    dominant = max(severity_by_domain, key=lambda d: _sev_idx(severity_by_domain[d]))
    overall = severity_by_domain[dominant]
    overall_profile = {
        "severity_by_domain": severity_by_domain,
        "dominant_area": dominant,
        "overall_severity": overall,
        "clinical_summary": "The profile reflects mixed performance across pragmatic and discourse functions.",
        "longitudinal_delta": None,
        "report_generated_at": "auto",
        "confidence": round(sum(d["confidence"] for d in domains.values()) / max(1, len(domains)), 3),
    }
    return domains, overall_profile
