# Call LLM

Makes up to three calls to the nominated LLM. The first two run at the same time; the feedback call runs after both:

- **Moderation prompt** (standalone) → returns a `passes_moderation` Boolean. If false, the response is marked incorrect with the message "Response did not pass moderation.", the correctness result is discarded and no feedback is generated.
- **Main prompt (`correctness_decision`)** + built-in JSON-output instruction → returns an `is_correct` Boolean
- **Main prompt (`correctness_decision`)** + **feedback prompt (`feedback_guidance`)**, told the correctness verdict → returns a `feedback` string. If this call fails, the correctness verdict is still returned, with the feedback "Could not evaluate the response, please try again."

Slow or failed calls are retried once while time allows. You can also set `fallback_models` to a list of model IDs that OpenRouter tries if the nominated model's provider returns an error.

If the LLM is still too slow or unavailable after retrying, the function returns a retryable error (HTTP 424, code `UPSTREAM_TIMEOUT` or `UPSTREAM_UNAVAILABLE`) with a "please try again" message instead of a generic failure. Configuration problems, such as an invalid API key or model, still return HTTP 500.

The `{{answer}}` field typically comes from Lambda Feedback's reference solution (the configure panel's `answer`). The `{{context}}` field is not automatically populated but can be added as a parameter, or just included directly in the prompt.
