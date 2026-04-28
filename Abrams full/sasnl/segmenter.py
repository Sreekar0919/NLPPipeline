from __future__ import annotations

from sasnl.models import Utterance, WordTiming


def _determine_speaker_roles(words: list[WordTiming]) -> dict[str, str]:
    """
    Determine speaker roles based on actual speaker IDs and word count.
    
    First checks if speaker roles are already specified in the WordTiming objects
    (e.g., parsed from [STUDENT]/[CLINICIAN] labels in the transcript).
    
    If not, the student typically speaks more in autism speaker assessment tasks,
    so we use word count to identify them.
    
    However, if speakers are explicitly labeled with indices (SPK_00, SPK_01, etc.),
    we assume the lower index is the interviewer and the higher index is the student.
    
    Returns a mapping of speaker_id -> speaker_role
    """
    # First check if roles are already specified in the words
    explicit_roles = {}
    for w in words:
        if w.speaker_role and w.speaker_id not in explicit_roles:
            explicit_roles[w.speaker_id] = w.speaker_role
    
    # If we found explicit roles for all speakers, use them
    if explicit_roles:
        return explicit_roles
    
    # Otherwise, determine roles based on speaker IDs and word count
    # Count words per speaker
    speaker_word_count: dict[str, int] = {}
    for w in words:
        speaker_word_count[w.speaker_id] = speaker_word_count.get(w.speaker_id, 0) + 1
    
    if not speaker_word_count:
        # Default fallback
        return {}
    
    # Check if speakers have numeric IDs (e.g., SPK_00, SPK_01)
    speaker_ids = list(speaker_word_count.keys())
    has_numeric_ids = all(
        id.startswith("SPK_") and id[4:].isdigit() 
        for id in speaker_ids
    )
    
    roles = {}
    
    if has_numeric_ids and len(speaker_ids) == 2:
        # If we have explicitly numbered speakers, use the numbering
        # Lower number = interviewer, higher number = student
        sorted_ids = sorted(speaker_ids, key=lambda x: int(x[4:]))
        roles[sorted_ids[0]] = "interviewer"  # SPK_00 -> interviewer
        roles[sorted_ids[1]] = "student"      # SPK_01 -> student
    else:
        # Otherwise use word count - speaker with most words is the student
        sorted_speakers = sorted(speaker_word_count.items(), key=lambda x: x[1], reverse=True)
        for idx, (speaker_id, _) in enumerate(sorted_speakers):
            if idx == 0:
                # Speaker with most words is the student
                roles[speaker_id] = "student"
            else:
                # Other speakers are interviewers
                roles[speaker_id] = "interviewer"
    
    return roles


def segment_words(words: list[WordTiming], session_duration_ms: int, gap_ms: int = 500) -> list[Utterance]:
    if not words:
        return []

    # Determine speaker roles based on actual speaker IDs
    speaker_roles = _determine_speaker_roles(words)

    utterances: list[Utterance] = []
    current: list[WordTiming] = [words[0]]

    def flush(turn_index: int, chunk: list[WordTiming]) -> Utterance:
        start_ms = chunk[0].start_ms
        end_ms = chunk[-1].end_ms
        speaker_id = chunk[0].speaker_id
        # Use the determined role, defaulting to "interviewer" if not found
        speaker_role = speaker_roles.get(speaker_id, "interviewer")
        text = " ".join(w.word for w in chunk)
        utt = Utterance(
            utterance_id=f"utt_{turn_index + 1:03}",
            speaker_id=speaker_id,
            speaker_role=speaker_role,
            start_ms=start_ms,
            end_ms=end_ms,
            turn_index=turn_index,
            text=text,
            words=chunk,
            word_count=len(chunk),
            start_time_s=start_ms / 1000.0,
            end_time_s=end_ms / 1000.0,
            t_norm=(start_ms / session_duration_ms) if session_duration_ms > 0 else 0.0,
        )
        return utt

    for w in words[1:]:
        prev = current[-1]
        speaker_changed = w.speaker_id != prev.speaker_id
        silence_gap = (w.start_ms - prev.end_ms) > gap_ms
        sentence_end = prev.word.endswith((".", "?", "!"))

        if speaker_changed or silence_gap or sentence_end:
            utterances.append(flush(len(utterances), current))
            current = [w]
        else:
            current.append(w)

    utterances.append(flush(len(utterances), current))
    return utterances
