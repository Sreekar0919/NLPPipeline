from __future__ import annotations

from collections import Counter
from typing import Any

from sasnl.models import Token, Utterance

FILLER_SINGLE = {"um", "uh", "like", "well"}
FILLER_MULTI = {"you know", "i mean"}


def _load_nlp(strict_mode: bool):
    try:
        import spacy  # type: ignore
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            nlp = spacy.blank("en")
        vader = SentimentIntensityAnalyzer()
        return nlp, vader
    except Exception as exc:  # noqa: BLE001
        if strict_mode:
            raise RuntimeError(
                "strict_mode requires spaCy and vaderSentiment; install with: pip install -e .[nlp]"
            ) from exc
        return None, None


def extract_features(utterances: list[Utterance], strict_mode: bool = True) -> dict:
    def _filler_hits(tokens: list[str]) -> list[str]:
        hits: list[str] = []
        i = 0
        while i < len(tokens):
            tk = tokens[i].lower()
            if tk in FILLER_SINGLE:
                hits.append(tk)
            if i + 1 < len(tokens):
                bi = f"{tokens[i].lower()} {tokens[i + 1].lower()}"
                if bi in FILLER_MULTI:
                    hits.append(bi)
            i += 1
        return hits

    nlp, vader = _load_nlp(strict_mode=strict_mode)

    for utt in utterances:
        tokens = utt.text.split()
        if nlp is not None:
            doc = nlp(utt.text)
            utt.spacy_doc = doc
            utt.tokens = [
                Token(index=i, text=t.text, lemma=t.lemma_.lower(), pos=t.pos_ or "X", dep=t.dep_ or "dep")
                for i, t in enumerate(doc)
            ]
        else:
            utt.tokens = [Token(index=i, text=t, lemma=t.lower(), pos="X", dep="dep") for i, t in enumerate(tokens)]
        filler_hits = _filler_hits(tokens)
        utt.audio_features = {
            "mean_F0": 150 + (utt.turn_index % 5) * 5,
            "intensity_mean": 0.5 + (utt.turn_index % 4) * 0.1,
            "speech_rate": 120 + (utt.turn_index % 6) * 10,
            "pause_before_ms": 800 if utt.turn_index else 0,
            "F0_slope": -0.2,
            "intensity_delta": -0.1,
        }
        utt.text_embedding = [0.0] * 384
        vader_score = vader.polarity_scores(utt.text)["compound"] if vader is not None else (0.1 if "fine" in utt.text.lower() else 0.0)
        utt.nlp_features = {
            "vader": {"compound": vader_score},
            "filler_count": len(filler_hits),
            "question": "?" in utt.text,
        }

    per_utterance = {
        u.utterance_id: {
            "speaker_role": u.speaker_role,
            "word_count": u.word_count,
            "sentence_count": 1,
            "sentence_types": ["simple"],
            "pos_counts": dict(Counter(t.pos for t in u.tokens)),
            "filler_hits": _filler_hits([t.text for t in u.tokens]),
            "false_start_hits": [],
            "repetition_hits": [],
            "question": u.nlp_features["question"],
            "morphology": {"verb_tenses": [], "agreement_errors": []},
        }
        for u in utterances
    }

    by_speaker: dict[str, list[Utterance]] = {}
    for u in utterances:
        by_speaker.setdefault(u.speaker_id, []).append(u)
    speaker_level = {}
    for spk, us in by_speaker.items():
        tokens = [t.text.lower() for u in us for t in u.tokens]
        unique = len(set(tokens))
        total = max(1, len(tokens))
        filler_total = sum(len(per_utterance[u.utterance_id]["filler_hits"]) for u in us)
        speaker_level[spk] = {
            "total_words": sum(u.word_count for u in us),
            "total_utterances": len(us),
            "avg_utterance_length_words": round(sum(u.word_count for u in us) / max(1, len(us)), 3),
            "ttr": round(unique / total, 4),
            "mattr_window_50": round(unique / total, 4),
            "unique_words": unique,
            "sentence_type_counts": {"simple": len(us)},
            "avg_sentence_length_words": round(sum(u.word_count for u in us) / max(1, len(us)), 3),
            "total_fillers": filler_total,
            "filler_rate_per_100_words": round(100 * filler_total / max(1, sum(u.word_count for u in us)), 3),
            "fillers_by_type": dict(Counter([h for u in us for h in per_utterance[u.utterance_id]["filler_hits"]])),
            "total_false_starts": 0,
            "total_repetitions": 0,
            "total_questions_asked": sum(1 for u in us if "?" in u.text),
            "total_turns": len(us),
        }

    turn_pairs = []
    for i in range(1, len(utterances)):
        a, b = utterances[i - 1], utterances[i]
        if a.speaker_role == "interviewer" and b.speaker_role == "student":
            turn_pairs.append(
                {
                    "turn_pair_id": f"tp_{len(turn_pairs) + 1:03}",
                    "interviewer_utterance_id": a.utterance_id,
                    "student_utterance_id": b.utterance_id,
                    "interviewer_word_count": a.word_count,
                    "student_word_count": b.word_count,
                    "interviewer_asked_question": "?" in a.text,
                    "student_asked_question": "?" in b.text,
                    "student_initiated": "?" not in a.text,
                }
            )

    student_words = sum(u.word_count for u in utterances if u.speaker_role == "student")
    interviewer_words = sum(u.word_count for u in utterances if u.speaker_role == "interviewer")
    full_transcript = {
        "total_words": sum(u.word_count for u in utterances),
        "total_turns": len(utterances),
        "turn_balance_ratio": round((student_words + 1) / max(1, interviewer_words + 1), 3),
        "word_balance_ratio": round(student_words / max(1, interviewer_words), 3),
        "interruptions": 0,
        "greeting_present": False,
        "departure_present": False,
        "topic_transition_count": 0,
        "session_duration_s": round(max((u.end_ms for u in utterances), default=0) / 1000.0, 3),
    }

    return {
        "computed_by": "phase_0_nlp",
        "tools_used": ["rule_based_mvp"],
        "per_utterance": per_utterance,
        "speaker_level": speaker_level,
        "turn_pairs": turn_pairs,
        "full_transcript": full_transcript,
    }
