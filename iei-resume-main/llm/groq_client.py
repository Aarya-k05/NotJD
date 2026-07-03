import requests
import json
import time
from typing import Optional

DEFAULT_GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions'


def call_groq(model: str, prompt: str, api_key: str, max_tokens: int = 1024, temperature: float = 0.0, attempts: int = 3, backoff_factor: float = 1.0, timeout: int = 30) -> str:
    """Call Groq's OpenAI-compatible chat completions API with retries.

    Attempts the request up to `attempts` times with exponential backoff.
    Returns the raw text output or raises RuntimeError after all retries fail.
    """
    if not api_key:
        raise RuntimeError('Groq call failed: no API key provided')

    url = DEFAULT_GROQ_URL
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'temperature': temperature,
    }

    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                if 'choices' in data and len(data['choices']) > 0:
                    c = data['choices'][0]
                    message = c.get('message') or {}
                    if isinstance(message, dict) and message.get('content'):
                        return message.get('content')
                    return c.get('text') or c.get('content') or json.dumps(c)
            return resp.text
        except Exception as e:
            last_exc = e
            if attempt < attempts:
                sleep_for = backoff_factor * (2 ** (attempt - 1))
                time.sleep(sleep_for)
                continue
            else:
                raise RuntimeError(f'Groq call failed after {attempts} attempts: {e}') from e
