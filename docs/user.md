# Call LLM

Makes up to three calls to the nominated LLM:

- **Moderation prompt** (standalone) → returns a `passes_moderation` Boolean. If false, evaluation stops here and skips the two calls below.
- **Main prompt (`correctness_decision`)** + built-in JSON-output instruction → returns an `is_correct` Boolean
- **Main prompt (`correctness_decision`)** + **feedback prompt (`feedback_guidance`)**, told the correctness verdict → returns a `feedback` string

The `{{answer}}` field typically comes from Lambda Feedback's reference solution (the configure panel's `answer`). The `{{context}}` field is not automatically populated but can be added as a parameter, or just included directly in the prompt.
