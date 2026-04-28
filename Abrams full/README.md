# SASNL Autism Speaker Study Tool

This repository implements a production-ready scaffold for the `.kiro` specification:
- audio-to-transcript pipeline
- utterance segmentation and temporal alignment
- feature extraction and prosody interpretation
- tiered agent battery (T1/T2/T3)
- Bedrock Claude-based LLM interpretation and narrative generation
- schema v1.1 session JSON output

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional heavy dependencies:

```bash
pip install -e .[audio,nlp,dev]
```

## AWS Bedrock setup

Set your AWS credentials/profile so `boto3` can call Bedrock.

Environment variables supported by this codebase:
- `AWS_PROFILE`
- `AWS_REGION` (default: `us-east-1`)

Model IDs are configurable in `sasnl/config.py`.

## Run

```bash
sasnl-run process ./combined_audio.wav --student-id student_001 --session-type SCS
```

### Transcript Support

You can also run the pipeline with a transcript instead of audio:

```bash
# From a transcript file (custom format)
sasnl-run process ./transcript.txt --student-id student_001 --session-type SCS

# From a JSON file
sasnl-run process ./transcript.json --student-id student_001 --session-type SCS --input-type transcript

# From a JSON string
sasnl-run process '[{"word": "hello", "start_ms": 0, "end_ms": 250, "speaker_id": "SPK_A"}]' --student-id student_001 --session-type SCS --input-type transcript

# From plain text
sasnl-run process 'Speaker A: hello there. Speaker B: how are you?' --student-id student_001 --session-type SCS --input-type transcript

# Auto-detect (uses file extension or format)
sasnl-run process ./transcript.txt --student-id student_001 --session-type SCS
```

**Supported transcript formats:**

1. **Custom sentence/timestamp format** (recommended):
   ```
   sentence: GO DO YOU HEAR [00:00:00,000 --> 00:00:01,700]
   timestamp: GO(0.000-0.680), DO(1.100-1.380), YOU(1.380-1.500), HEAR(1.500-1.700)
   
   sentence: BUT IN LESS THAN FIVE MINUTES [00:00:02,320 --> 00:00:05,840]
   timestamp: BUT(2.320-2.480), IN(2.480-2.600), LESS(2.600-2.760), THAN(2.760-2.920), FIVE(2.920-3.140), MINUTES(3.140-3.380)
   ```
   - Each sentence line shows the full text and timing range
   - Followed by a timestamp line with individual word timings in `WORD(start-end)` format
   - Times are in seconds, automatically converted to milliseconds
   - Speaker roles alternate by sentence (SPK_A, SPK_B, SPK_A, ...)

2. **JSON list format**:
   ```json
   [
     {"word": "hello", "start_ms": 0, "end_ms": 250, "speaker_id": "SPK_A"},
     {"word": "there", "start_ms": 300, "end_ms": 550, "speaker_id": "SPK_A"}
   ]
   ```

3. **JSON with utterances**:
   ```json
   {
     "utterances": [
       {"word": "hello", "start_ms": 0, "end_ms": 250, "speaker_id": "SPK_A"}
     ]
   }
   ```

4. **Plain text** (limited - no timing):
   ```
   Speaker A: hello there
   Speaker B: how are you?
   ```

By default, output is written to `./outputs/session_<timestamp>.json`.
# NLPPipeline
