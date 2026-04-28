from __future__ import annotations

from sasnl.models import Utterance


MENTAL_STATE_VERBS = {"think", "know", "believe", "remember", "forget", "wonder", "guess", "understand"}
CAUSAL_CUES = {"because", "so", "therefore", "since", "if", "then", "but", "however", "although"}
IDIOM_CUES = {"piece of cake", "break a leg", "spill the beans", "under the weather"}
HEDGE_CUES = {"maybe", "i think", "kind of", "sort of"}


def has_filled_pause(text: str) -> bool:
    t = text.lower()
    return any(x in t for x in [" um ", " uh ", "...", "erm", "hmm"])


def has_repair_signal(text: str) -> bool:
    t = text.lower()
    return any(x in t for x in ["i mean", "sorry", "no wait", "actually", "let me"])


def has_fragment_signal(text: str) -> bool:
    t = text.strip().lower()
    return len(t.split()) < 4 or t.endswith(("...", "-"))


def has_logic_cue(text: str) -> bool:
    return any(cue in text.lower().split() for cue in CAUSAL_CUES)


def has_idiom_signal(text: str) -> bool:
    low = text.lower()
    return any(cue in low for cue in IDIOM_CUES)


def has_question_signal(text: str) -> bool:
    low = text.lower()
    return "?" in text or low.startswith(("what", "why", "how", "when", "where", "who", "do ", "did "))


def has_long_noun_chunk_proxy(text: str) -> bool:
    return max((len(c.split()) for c in text.split(",")), default=0) > 8


def has_mental_state_verb(text: str) -> bool:
    return any(v in text.lower().split() for v in MENTAL_STATE_VERBS)


def has_narrative_segment(utterances: list[Utterance]) -> bool:
    if not utterances:
        return False
    joined = " ".join(u.text.lower() for u in utterances)
    return any(cue in joined for cue in ["once", "then", "after", "finally", "yesterday"])


def mismatch_score(prosodic_valence: float, text_valence: float) -> float:
    return abs(prosodic_valence - text_valence) / 2.0
