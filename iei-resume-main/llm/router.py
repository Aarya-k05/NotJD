from typing import Optional, Callable

from llm.groq_client import call_groq
from llm.ollama_client import call_ollama


def call_llm(
    provider: str,
    prompt: str,
    *,
    groq_model: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    ollama_model: Optional[str] = None,
    ollama_host: Optional[str] = None,
    attempts: int = 3,
    backoff_factor: float = 1.0,
    fallback_to_ollama: bool = True,
    log: Optional[Callable[[str], None]] = None,
) -> str:
    """Route an LLM call to the configured provider.

    - provider == 'groq': try Groq first. If it fails and `fallback_to_ollama`
      is True, log a warning (via the optional `log` callback, following the
      same pattern as main.py's `log()` closure) and retry with Ollama.
    - provider == 'ollama': call Ollama directly (already the base case, no
      fallback needed).

    Returns the raw text output, matching the contract of `call_ollama` /
    `call_groq`. Raises RuntimeError if all applicable attempts fail.
    """
    if provider == 'groq':
        try:
            return call_groq(
                groq_model,
                prompt,
                api_key=groq_api_key,
                attempts=attempts,
                backoff_factor=backoff_factor,
            )
        except Exception as e:
            if fallback_to_ollama:
                if log:
                    log(f'Groq call failed, falling back to Ollama: {str(e)[:100]}')
                return call_ollama(
                    ollama_model,
                    prompt,
                    host=ollama_host,
                    attempts=attempts,
                    backoff_factor=backoff_factor,
                )
            raise
    elif provider == 'ollama':
        return call_ollama(
            ollama_model,
            prompt,
            host=ollama_host,
            attempts=attempts,
            backoff_factor=backoff_factor,
        )
    else:
        raise ValueError(f'Unknown LLM provider: {provider}')
