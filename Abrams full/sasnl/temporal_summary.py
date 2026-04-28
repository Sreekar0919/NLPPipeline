from __future__ import annotations

from sasnl.models import Utterance


TRACKED_AGENTS = ["EmpathyAgent", "SarcasmDetectionAgent", "TopicManagementAgent", "ExecutiveFunctionAgent"]


def _chunk(values: list[Utterance], parts: int = 4) -> list[list[Utterance]]:
    if not values:
        return [[] for _ in range(parts)]
    n = len(values)
    chunk = max(1, n // parts)
    out = []
    i = 0
    while i < n and len(out) < parts - 1:
        out.append(values[i : i + chunk])
        i += chunk
    out.append(values[i:])
    while len(out) < parts:
        out.append([])
    return out


def build_temporal_summary(utterances: list[Utterance], agent_outputs: dict) -> dict:
    quarters = _chunk(sorted(utterances, key=lambda u: u.turn_index), 4)
    agent_arcs: dict[str, list[float]] = {}
    for name in TRACKED_AGENTS:
        conf = float(agent_outputs.get(name).interpretation.get("confidence", 0.0)) if name in agent_outputs else 0.0
        agent_arcs[name] = [conf, conf, conf, conf]

    intensity_arc = []
    flag_density_arc = []
    for q in quarters:
        if q:
            intensity_arc.append(sum(float(u.audio_features.get("intensity_mean", 0.0)) for u in q) / len(q))
            flag_density_arc.append(sum(len(u.clinical_flags) for u in q))
        else:
            intensity_arc.append(0.0)
            flag_density_arc.append(0)

    return {
        "quartiles": 4,
        "agent_arcs": agent_arcs,
        "prosodic_arc_intensity_mean": intensity_arc,
        "flag_density_arc": flag_density_arc,
    }
