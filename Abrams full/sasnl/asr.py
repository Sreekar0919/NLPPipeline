from __future__ import annotations

import json
from pathlib import Path

from sasnl.models import WordTiming


def _fallback_transcript() -> list[WordTiming]:
    words = [
        "hello",
        "there",
        "how",
        "are",
        "you",
        "today",
        "i",
        "am",
        "doing",
        "fine",
    ]
    out: list[WordTiming] = []
    t = 0
    for idx, w in enumerate(words):
        speaker = "SPK_A" if idx < 6 else "SPK_B"
        out.append(WordTiming(word=w, start_ms=t, end_ms=t + 250, speaker_id=speaker))
        t += 300
    return out


def parse_transcript(transcript_input: str | dict) -> list[WordTiming]:
    """
    Parse a transcript from either a JSON string, file path, or dict.
    
    Supported formats:
    - JSON file with list of word objects: [{"word": "hello", "start_ms": 0, "end_ms": 250, "speaker_id": "SPK_A"}]
    - JSON file with utterances: {"utterances": [...]}
    - Custom format with sentence and word timings:
        [SPEAKER 1] GO DO YOU HEAR [00:00:00,000 --> 00:00:01,700]
        timestamp: GO(0.000-0.680), DO(1.100-1.380), YOU(1.380-1.500), HEAR(1.500-1.700)
    - Plain text: "Speaker A: hello there. Speaker B: how are you?"
    """
    # If it's already a dict, use it directly
    if isinstance(transcript_input, dict):
        data = transcript_input
    else:
        # Try to parse as JSON first
        try:
            data = json.loads(transcript_input)
        except json.JSONDecodeError:
            # Check if it looks like a file path (simple heuristic: no newlines and ends with .json or .txt)
            if "\n" not in transcript_input and (transcript_input.endswith(".json") or transcript_input.endswith(".txt")):
                path = Path(transcript_input)
                if path.exists() and path.is_file():
                    with open(path) as f:
                        content = f.read()
                        # Try JSON first
                        try:
                            data = json.loads(content)
                        except json.JSONDecodeError:
                            # Fall back to custom format parsing
                            return _parse_custom_transcript_format(content)
            
            # Try custom format parsing for multi-line text or plain text
            if "[SPEAKER" in transcript_input or "\ntimestamp:" in transcript_input:
                return _parse_custom_transcript_format(transcript_input)
            
            # Fall back to parsing as plain text
            return _parse_plain_text_transcript(transcript_input)
    
    # If data is a dict with utterances key, extract the list
    if isinstance(data, dict) and "utterances" in data:
        words_data = data["utterances"]
    elif isinstance(data, list):
        words_data = data
    else:
        raise ValueError("Transcript must be a list of word objects or dict with 'utterances' key")
    
    # Convert to WordTiming objects
    out: list[WordTiming] = []
    for item in words_data:
        if isinstance(item, dict) and "word" in item:
            out.append(
                WordTiming(
                    word=item["word"],
                    start_ms=int(item.get("start_ms", 0)),
                    end_ms=int(item.get("end_ms", 0)),
                    speaker_id=item.get("speaker_id", "SPK_A"),
                )
            )
    
    if not out:
        raise ValueError("No valid word entries found in transcript")
    return out


def _parse_custom_transcript_format(text: str) -> list[WordTiming]:
    """
    Parse custom transcript format with sentence and word timings:
    
    sentence: GO DO YOU HEAR [00:00:00,000 --> 00:00:01,700]
    timestamp: GO(0.000-0.680), DO(1.100-1.380), YOU(1.380-1.500), HEAR(1.500-1.700)
    
    Or with speaker format:
    [SPEAKER 1] GO DO YOU HEAR [00:00:00,000 --> 00:00:01,700]
    timestamp: GO(0.000-0.680), DO(1.100-1.380), YOU(1.380-1.500), HEAR(1.500-1.700)
    
    [SPEAKER 00] sentence text [timecode]
    timestamp: WORD(start-end), WORD(start-end), ...
    
    Or with role labels:
    sentence: [STUDENT] text here [timecode]
    timestamp: WORD(start-end), WORD(start-end), ...
    
    sentence: [CLINICIAN] text here [timecode]
    timestamp: WORD(start-end), WORD(start-end), ...
    """
    import re
    
    out: list[WordTiming] = []
    lines = text.strip().split('\n')
    
    i = 0
    speaker_counter = 1  # For alternating speakers when not explicitly labeled
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        speaker_id = None
        speaker_role = None
        
        # Check if this is a sentence line (starts with "sentence:" or "[SPEAKER")
        if line.startswith("sentence:"):
            # Extract the sentence text and look for speaker labels
            line_content = line[9:].strip()  # Remove "sentence: "
            
            # Try to extract role labels like [STUDENT] or [CLINICIAN]
            role_match = re.search(r'^\[?(STUDENT|CLINICIAN)\]?', line_content, re.IGNORECASE)
            if role_match:
                role_text = role_match.group(1).lower()
                speaker_role = "student" if role_text == "student" else "interviewer"
                # Remove the role label from content
                line_content = re.sub(r'^\[?(STUDENT|CLINICIAN)\]?\s*', '', line_content, flags=re.IGNORECASE)
            
            # Try to extract speaker number from the sentence if present
            # e.g., "[SPEAKER 01] some text" or "01: some text"
            speaker_match = re.search(r'^\[?SPEAKER\s+(\d+)\]?|^(\d+):', line_content, re.IGNORECASE)
            if speaker_match:
                speaker_num_val = int(speaker_match.group(1) or speaker_match.group(2))
                speaker_id = f"SPK_{speaker_num_val:02d}"
            else:
                # No explicit speaker number, alternate between speakers
                speaker_id = "SPK_A" if speaker_counter % 2 == 1 else "SPK_B"
                speaker_counter += 1
                
        elif line.startswith("[SPEAKER"):
            # Extract speaker number from [SPEAKER n] where n can be 00, 01, 1, 2, etc.
            speaker_match = re.search(r'\[SPEAKER\s+(\d+)\]', line, re.IGNORECASE)
            if speaker_match:
                speaker_num_val = int(speaker_match.group(1))
                # Use the actual speaker number as the ID: SPEAKER 00 -> SPK_00, SPEAKER 01 -> SPK_01, etc.
                # This preserves the original speaker labels from the transcript
                speaker_id = f"SPK_{speaker_num_val:02d}"
            line_content = line.split("]", 1)[1].strip() if "]" in line else line
        else:
            i += 1
            continue
        
        # Look for timestamp line on next line
        timestamp_line = ""
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line.lower().startswith("timestamp:"):
                timestamp_line = next_line
                i += 1
        
        # Parse word timings from timestamp line
        if timestamp_line and speaker_id:
            # Format: timestamp: WORD(start-end), WORD(start-end), ...
            word_pattern = r'(\w+\'?\w*)\(([0-9.]+)-([0-9.]+)\)'
            matches = re.findall(word_pattern, timestamp_line)
            
            for word, start_sec, end_sec in matches:
                # Convert seconds to milliseconds
                start_ms = int(float(start_sec) * 1000)
                end_ms = int(float(end_sec) * 1000)
                
                out.append(
                    WordTiming(
                        word=word.lower(),
                        start_ms=start_ms,
                        end_ms=end_ms,
                        speaker_id=speaker_id,
                        speaker_role=speaker_role,
                    )
                )
        
        i += 1
    
    if not out:
        # If custom format didn't work, try plain text parsing
        return _parse_plain_text_transcript(text)
    
    return out


def _parse_plain_text_transcript(text: str) -> list[WordTiming]:
    """
    Parse a simple plain text transcript like:
    "Speaker A: hello there. Speaker B: how are you?"
    """
    out: list[WordTiming] = []
    current_speaker = "SPK_A"
    current_time = 0
    
    # Split by lines
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for speaker prefixes like "Speaker A:", "SPK_A:", "A:", etc.
        for prefix in ["Speaker A:", "Speaker B:", "SPK_A:", "SPK_B:", "A:", "B:", "SPEAKER 1:", "SPEAKER 2:"]:
            if line.startswith(prefix):
                if "1" in prefix or "A" in prefix:
                    current_speaker = "SPK_A"
                else:
                    current_speaker = "SPK_B"
                line = line[len(prefix):].strip()
                break
        
        # Split into words
        words = line.split()
        for word in words:
            word = word.strip('.,!?;:"')
            if word:
                out.append(
                    WordTiming(
                        word=word.lower(),
                        start_ms=current_time,
                        end_ms=current_time + 250,
                        speaker_id=current_speaker,
                    )
                )
                current_time += 300
    
    if not out:
        raise ValueError("No words found in plain text transcript")
    return out


def transcribe_audio(audio_path: Path, strict_mode: bool = True) -> list[WordTiming]:
    if not audio_path.exists() or not audio_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist or is unreadable: {audio_path}")

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as exc:  # noqa: BLE001
        if strict_mode:
            raise RuntimeError(
                "strict_mode requires faster-whisper for real ASR word timestamps; "
                "install with: pip install -e .[audio]"
            ) from exc
        return _fallback_transcript()

    model = WhisperModel("large-v3", compute_type="int8")
    segments, _ = model.transcribe(str(audio_path), word_timestamps=True)
    out: list[WordTiming] = []
    spk = "SPK_A"
    for segment in segments:
        if not segment.words:
            continue
        for w in segment.words:
            if w.start is None or w.end is None:
                continue
            out.append(
                WordTiming(
                    word=w.word.strip(),
                    start_ms=int(w.start * 1000),
                    end_ms=int(w.end * 1000),
                    speaker_id=spk,
                )
            )
        # simple role alternation heuristic until diarization is integrated.
        spk = "SPK_B" if spk == "SPK_A" else "SPK_A"

    if not out:
        raise RuntimeError("ASR returned no words.")
    return out
