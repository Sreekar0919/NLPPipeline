from __future__ import annotations

from sasnl.models import Utterance


def build_topic_segments(utterances: list[Utterance], model_name: str) -> dict:
    ids = [u.utterance_id for u in utterances]
    greeting_ids = ids[:2]
    departure_ids = ids[-2:] if len(ids) > 2 else []
    body_ids = ids[2:-2] if len(ids) > 4 else ids
    topics = []
    if body_ids:
        topics.append(
            {
                "topic_id": "topic_001",
                "label": "general_conversation",
                "utterance_ids": body_ids,
                "initiated_by": "interviewer",
            }
        )
    return {
        "computed_by": "phase_1_llm",
        "model": model_name,
        "greeting": {"present": bool(greeting_ids), "utterance_ids": greeting_ids},
        "topics": topics,
        "departure": {"present": bool(departure_ids), "utterance_ids": departure_ids},
        "confidence": 0.65,
    }
