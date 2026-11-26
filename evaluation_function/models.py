"""Model tiers: stable names that stand in for specific OpenRouter models.

Tasks can set `model` to a tier name instead of an OpenRouter ID, so retiring
a model means editing this file and redeploying rather than updating every
task that used it.

Each tier lists its primary model first, followed by built-in fallbacks. Each
retry after a timeout, provider error or missing model moves to the next one
(see evaluation._request_json). When updating a tier:

- Every model must have a Zero Data Retention endpoint that supports
  `response_format`; check with
  `curl -s https://openrouter.ai/api/v1/endpoints/zdr | jq -r '.data[].model_id' | sort -u`.
- The fallback should come from a different provider than the primary, so a
  provider outage doesn't take out the whole tier.
- A tier may set `reasoning`, passed through as OpenRouter's `reasoning`
  request field. Keep the effort low: every attempt is capped at 8s.
"""

MODEL_TIERS = {
    "small": {
        "models": ["openai/gpt-6-luna", "anthropic/claude-haiku-5.5"],
    },
    "medium": {
        "models": ["google/gemini-3.8-flash", "openai/gpt-6.1-terra"],
    },
    "large": {
        "models": ["anthropic/claude-sonnet-5.5", "openai/gpt-6.2-sol"],
    },
    "reasoning": {
        "models": ["openai/gpt-6.2-sol", "anthropic/claude-sonnet-5.5"],
        "reasoning": {"effort": "low"},
    },
}

DEFAULT_TIER = "medium"


def resolve_models(model, fallback_models=None):
    """The models to try for model, in order: a tier's models, or model itself,
    followed by fallback_models (ignored unless it is a list of strings)."""
    tier = MODEL_TIERS.get(model)
    models = list(tier["models"]) if tier else [model]
    if isinstance(fallback_models, list) and all(isinstance(m, str) for m in fallback_models):
        models += fallback_models
    return models
