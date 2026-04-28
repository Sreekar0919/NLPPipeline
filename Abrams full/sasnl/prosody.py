from __future__ import annotations

import math

from sasnl.models import Utterance


def _z(value: float, mean: float, std: float) -> float:
    if std == 0:
        return 0.0
    return (value - mean) / std


def compute_session_baseline(utterances: list[Utterance]) -> dict:
    student = [u for u in utterances if u.speaker_role == "student"] or utterances

    def mean_std(key: str) -> tuple[float, float]:
        vals = [float(u.audio_features.get(key, 0.0)) for u in student]
        if not vals:
            return 0.0, 0.0
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return mean, math.sqrt(var)

    f0m, f0s = mean_std("mean_F0")
    im, istd = mean_std("intensity_mean")
    rm, rs = mean_std("speech_rate")
    pm, ps = mean_std("pause_before_ms")
    return {
        "F0_mean": f0m,
        "F0_std": f0s,
        "intensity_mean": im,
        "intensity_std": istd,
        "rate_mean": rm,
        "rate_std": rs,
        "pause_mean": pm,
        "pause_std": ps,
    }


def interpret_prosody(utterances: list[Utterance], baseline: dict, pause_flag_ms: int = 1500) -> None:
    for u in utterances:
        obs: list[str] = []
        f0z = _z(u.audio_features.get("mean_F0", 0.0), baseline["F0_mean"], baseline["F0_std"])
        iz = _z(
            u.audio_features.get("intensity_mean", 0.0),
            baseline["intensity_mean"],
            baseline["intensity_std"],
        )
        rz = _z(u.audio_features.get("speech_rate", 0.0), baseline["rate_mean"], baseline["rate_std"])
        if abs(f0z) > 1.5:
            obs.append("Pitch deviates from session baseline")
        if abs(iz) > 1.5:
            obs.append("Intensity deviates from session baseline")
        if abs(rz) > 1.5:
            obs.append("Speech rate deviates from session baseline")
        if u.audio_features.get("pause_before_ms", 0) > pause_flag_ms:
            obs.append("Long pause before utterance")
        u.prosody_text = ". ".join(obs) if obs else "Prosody within normal range for this student."
