# Setup Instructions for Bedrock API Key

## Quick Setup

1. **Add your API key to `.env`** (already created in project root):

```bash
# .env file (in project root directory)
BEDROCK_API_KEY=your_actual_api_key_here
BEDROCK_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0
```

2. **Replace the placeholder**:
   - Open `.env` file
   - Replace `your_bedrock_api_key_here` with your actual Bedrock API key (Bearer token)

3. **Run the pipeline** normally:

```bash
sasnl-run --input-source ./transcript.txt --student-id student_001 --session-type SCS --no-strict
```

## How It Works

The system now uses:
- ✅ **Bearer token authentication** (simple HTTP requests)
- ✅ **No AWS secret keys needed**
- ✅ **Environment-based configuration** via `.env` file
- ✅ **API key is never committed to Git** (added to `.gitignore`)

## Key Files Updated

- **`sasnl/llm.py`**: Now uses `requests` library with Bearer token instead of boto3
- **`.env`**: Contains your API credentials (not tracked by Git)
- **`pyproject.toml`**: Removed `boto3`, added `requests` and `python-dotenv`
- **`.gitignore`**: Prevents `.env` from being committed

## Important Notes

⚠️ **Security**:
- Never commit the `.env` file
- Keep your API key secret
- The `.env` file is automatically excluded from Git

## Model IDs You Can Use

Replace `BEDROCK_MODEL_ID` with any of these:
- `anthropic.claude-3-5-sonnet-20240620-v1:0` (current)
- `anthropic.claude-3-haiku-20240307-v1:0`
- `meta.llama3-70b-instruct-v1:0`
- `amazon.titan-text-express-v1`

## Troubleshooting

If you get an error about missing API key:
1. Verify `.env` file exists in project root
2. Check `BEDROCK_API_KEY` is set to your actual key
3. Make sure the key hasn't expired
4. Try running with `--no-strict` flag for testing

## API Response Example

The system expects Claude responses in this format:
```json
{
  "content": [
    {"text": "...response text..."}
  ]
}
```

Everything else is handled automatically by the client.
