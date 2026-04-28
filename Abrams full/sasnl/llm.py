from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from sasnl.config import ModelConfig

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Get Bedrock API configuration from environment
BEDROCK_API_KEY = os.getenv("BEDROCK_API_KEY", "")
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")


@dataclass
class BedrockClaudeClient:
    config: ModelConfig
    region_name: str = "us-east-1"
    strict_mode: bool = True

    def __post_init__(self) -> None:
        # Validate API key is set
        if not BEDROCK_API_KEY or BEDROCK_API_KEY == "your_bedrock_api_key_here":
            if self.strict_mode:
                raise RuntimeError(
                    "strict_mode requires valid BEDROCK_API_KEY. "
                    "Set BEDROCK_API_KEY in .env file."
                )
        
        self.api_key = BEDROCK_API_KEY
        self.region = BEDROCK_REGION

    def _validate_credentials(self) -> None:
        """Validate that API key is available."""
        if not self.api_key or self.api_key == "your_bedrock_api_key_here":
            raise RuntimeError(
                "strict_mode requires valid BEDROCK_API_KEY. "
                "Set BEDROCK_API_KEY in .env file."
            )

    def invoke_json(self, prompt: str, system_prompt: str, narrator: bool = False) -> dict:
        """
        Invoke Claude model via Stanford Healthcare Bedrock API.
        Uses api-key authentication.
        """
        model_id = self.config.narrator_model_id if narrator else self.config.analyst_model_id
        
        # Construct the Stanford Healthcare API endpoint URL
        # Note: model_id format should be like: us.anthropic.claude-haiku-4-5-20251001-v1:0
        url = f"https://aihubapi.stanfordhealthcare.org/aws-bedrock/model/{model_id}/invoke"
        
        # Prepare headers with api-key authentication
        headers = {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "api-key": self.api_key,
        }
        
        # Prepare the payload for Claude
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }
        
        try:
            # Encode data
            data = json.dumps(payload)
            req = urllib.request.Request(
                url, 
                headers=headers, 
                data=bytes(data.encode("utf-8"))
            )
            req.get_method = lambda: 'POST'
            
            # Make the request
            response = urllib.request.urlopen(req)
            response_data = response.read().decode("utf-8")
            result = json.loads(response_data)
            
            # Extract text from response
            text = result.get("content", [{}])[0].get("text", "")
            
            if not text:
                raise ValueError("No text in response")
            
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw_text": text}
                
        except urllib.error.HTTPError as exc:
            # Handle HTTP errors
            if self.strict_mode:
                error_body = exc.read().decode("utf-8") if hasattr(exc, 'read') else str(exc)
                raise RuntimeError(f"API invocation failed in strict_mode: {error_body}") from exc
            
            if narrator:
                return {
                    "narrative": (
                        "Fluency observations were collected with mild disfluency markers. "
                        "Language form appeared variable across turns. Pragmatic performance showed mixed reciprocal response quality. "
                        "Prosody-pragmatics alignment was interpreted conservatively due to runtime fallback mode."
                    )
                }
            return {
                "functional_label": "fallback_mode",
                "severity": "mild",
                "clinical_note": f"LLM fallback used due to API error: {exc}",
                "confidence": 0.5,
            }
        except Exception as exc:
            # Handle other errors
            if self.strict_mode:
                raise RuntimeError(f"Bedrock API invocation failed in strict_mode: {exc}") from exc
            
            if narrator:
                return {
                    "narrative": (
                        "Fluency observations were collected with mild disfluency markers. "
                        "Language form appeared variable across turns. Pragmatic performance showed mixed reciprocal response quality. "
                        "Prosody-pragmatics alignment was interpreted conservatively due to runtime fallback mode."
                    )
                }
            return {
                "functional_label": "fallback_mode",
                "severity": "mild",
                "clinical_note": f"LLM fallback used due to runtime error: {exc}",
                "confidence": 0.5,
            }
