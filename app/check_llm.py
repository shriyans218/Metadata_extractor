"""Quick sanity check: python -m app.check_llm  -- prints the exact reason if the LLM call fails."""
from dotenv import load_dotenv
load_dotenv(override=True)
from openai import OpenAI
from app.extraction.extractor import _provider_config, _call_llm, LLMError
try:
    base, key, model = _provider_config()
    print("provider url:", base, "| model:", model)
    print(_call_llm(OpenAI(base_url=base, api_key=key), model, 'Return {"ok": true} as JSON.', 0.0, retries=1))
except LLMError as e:
    print("FAILED:", e)
