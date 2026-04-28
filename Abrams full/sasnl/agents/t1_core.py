from __future__ import annotations

from collections import Counter

from sasnl.agents.base import Agent
from sasnl.models import EvidenceItem, Utterance

FILLER_LEXICON = {"um", "uh", "like", "you know", "well", "i mean"}


class FillerWordAgent(Agent):
    name = "FillerWordAgent"

    def run(self, context: dict):
        utterances: list[Utterance] = context["student_utterances"]
        tokens = [t.text.lower() for u in utterances for t in u.tokens]
        counts = Counter(t for t in tokens if t in {"um", "uh", "like", "well"})
        total_fillers = sum(counts.values())
        total_tokens = max(len(tokens), 1)
        evidence = []
        for u in utterances:
            for t in u.tokens:
                if t.text.lower() in {"um", "uh", "like", "well"}:
                    evidence.append(
                        EvidenceItem(
                            evidence_type="token_instance",
                            utterance_id=u.utterance_id,
                            utterance_text=u.text,
                            start_ms=u.start_ms,
                            end_ms=u.end_ms,
                            surface=t.text,
                            feature_type=t.text.lower(),
                        )
                    )
        return self._output(
            {
                "total_fillers": total_fillers,
                "filler_rate_per_100_words": round(100 * total_fillers / total_tokens, 2),
                "fillers_by_type": dict(counts),
            },
            evidence=evidence,
        )


class RepetitionAgent(Agent):
    name = "RepetitionAgent"

    def run(self, context: dict):
        utterances: list[Utterance] = context["student_utterances"]
        repeats = 0
        for i in range(1, len(utterances)):
            a = set(utterances[i - 1].text.lower().split())
            b = set(utterances[i].text.lower().split())
            overlap = len(a & b) / max(1, len(a | b))
            if overlap > 0.5:
                repeats += 1
        return self._output({"repetition_count": repeats})


class SpeechRateRhythmAgent(Agent):
    name = "SpeechRateRhythmAgent"

    def run(self, context: dict):
        utterances: list[Utterance] = context["student_utterances"]
        wpms = []
        for u in utterances:
            dur_min = max((u.end_ms - u.start_ms) / 60000.0, 1e-3)
            wpms.append(u.word_count / dur_min)
        mean_wpm = sum(wpms) / max(1, len(wpms))
        flag = mean_wpm < 80 or mean_wpm > 200
        return self._output({"mean_wpm": round(mean_wpm, 2), "clinical_flag": flag})
