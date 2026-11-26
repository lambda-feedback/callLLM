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
- In MODEL_TIERS, the fallback should come from a different provider than the
  primary, so a provider outage doesn't take out the whole tier.
- In PROVIDER_TIERS, every model must come from that provider: these tiers
  exist for tasks that want one provider's style consistently, so they
  deliberately don't fall back to another provider.
- A tier may set `reasoning`, passed through as OpenRouter's `reasoning`
  request field. Keep the effort low: every attempt is capped at 8s.

Tasks pick a provider tier with `provider` alongside `model`, e.g.
`{"model": "medium", "provider": "openai"}`. To add a provider, add a key to
PROVIDER_TIERS covering every tier name in MODEL_TIERS.
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

PROVIDER_TIERS = {
    "openai": {
        "small": {"models": ["openai/gpt-6-luna", "openai/gpt-5.6-luna"]},
        "medium": {"models": ["openai/gpt-6.1-terra", "openai/gpt-5.6-terra"]},
        "large": {"models": ["openai/gpt-6.2-sol", "openai/gpt-6.1-sol"]},
        "reasoning": {
            "models": ["openai/gpt-6.2-sol", "openai/gpt-6.1-sol"],
            "reasoning": {"effort": "low"},
        },
    },
    "google": {
        "small": {"models": ["google/gemini-3.5-flash-lite", "google/gemini-3.1-flash-lite"]},
        "medium": {"models": ["google/gemini-3.8-flash", "google/gemini-3.7-flash"]},
        "large": {"models": ["google/gemini-4-carbon", "google/gemini-4-hydrogen"]},
        "reasoning": {
            "models": ["google/gemini-4-argon", "google/gemini-4-carbon"],
            "reasoning": {"effort": "low"},
        },
    },
    "anthropic": {
        "small": {"models": ["anthropic/claude-haiku-5.5", "anthropic/claude-haiku-4.5"]},
        "medium": {"models": ["anthropic/claude-sonnet-5.5", "anthropic/claude-sonnet-5"]},
        "large": {"models": ["anthropic/claude-opus-5.5", "anthropic/claude-opus-5"]},
        "reasoning": {
            "models": ["anthropic/claude-sonnet-5.5", "anthropic/claude-sonnet-5"],
            "reasoning": {"effort": "low"},
        },
    },
    "deepseek": {
        "small": {"models": ["deepseek/deepseek-v4.1-flash", "deepseek/deepseek-v4-flash"]},
        # DeepSeek has no model above V4 Pro, so large repeats medium.
        "medium": {"models": ["deepseek/deepseek-v4-pro-0813", "deepseek/deepseek-v4-pro"]},
        "large": {"models": ["deepseek/deepseek-v4-pro-0813", "deepseek/deepseek-v4-pro"]},
        "reasoning": {
            "models": ["deepseek/deepseek-v4-pro-0813", "deepseek/deepseek-r1-0528"],
            "reasoning": {"effort": "low"},
        },
    },
}

DEFAULT_TIER = "medium"


def get_tier(model, provider=None):
    """The tier definition model names, or None if model isn't a tier name.

    With provider, a tier name resolves in that provider's tiers; an unknown
    provider raises ValueError. provider is ignored for explicit model IDs,
    which already name their provider."""
    if model not in MODEL_TIERS:
        return None
    if provider is None:
        return MODEL_TIERS[model]
    if provider not in PROVIDER_TIERS:
        raise ValueError(
            f"unknown provider {provider!r}; expected one of {sorted(PROVIDER_TIERS)}"
        )
    return PROVIDER_TIERS[provider][model]


def resolve_models(model, fallback_models=None, provider=None):
    """The models to try for model, in order: a tier's models, or model itself,
    followed by fallback_models (ignored unless it is a list of strings)."""
    tier = get_tier(model, provider)
    models = list(tier["models"]) if tier else [model]
    if isinstance(fallback_models, list) and all(isinstance(m, str) for m in fallback_models):
        models += fallback_models
    return models
