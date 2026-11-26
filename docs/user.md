# Call LLM

Makes up to three calls to the nominated LLM. The first two run at the same time; the feedback call runs after both:

- **Moderation prompt** (standalone) → returns a `passes_moderation` Boolean. If false, the response is marked incorrect with the message "Response did not pass moderation.", the correctness result is discarded and no feedback is generated.
- **Main prompt (`correctness_decision`)** + built-in JSON-output instruction → returns an `is_correct` Boolean
- **Main prompt (`correctness_decision`)** + **feedback prompt (`feedback_guidance`)**, told the correctness verdict → returns a `feedback` string. If this call fails, the correctness verdict is still returned, with the feedback "Could not evaluate the response, please try again."

The model can be a specific OpenRouter model ID or one of the tiers `small`, `medium`, `large` or `reasoning`. Each tier points to a current model (plus a backup from another provider), and the maintainers update it when models are retired, so a tier is the safest choice for long-lived tasks. If no model is given, `medium` is used.

Different providers give noticeably different feedback. To stick with one, add `provider` (`openai`, `google`, `anthropic` or `deepseek`) next to a tier, e.g. `"model": "medium", "provider": "openai"`. The tier's backup then comes from the same provider too.

Slow or failed calls are retried while time allows. Each retry uses the next backup model: a tier's built-in backup first, then any model IDs you list in `fallback_models`.

If the LLM is still too slow or unavailable after retrying, the function returns a retryable error (HTTP 424, code `UPSTREAM_TIMEOUT` or `UPSTREAM_UNAVAILABLE`) with a "please try again" message instead of a generic failure. Configuration problems, such as an invalid API key or model, still return HTTP 500.

The `{{answer}}` field typically comes from Lambda Feedback's reference solution (the configure panel's `answer`). The `{{context}}` field is not automatically populated but can be added as a parameter, or just included directly in the prompt.
