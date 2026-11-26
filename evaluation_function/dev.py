import argparse
import json

from .evaluation import evaluation_function

DEFAULT_PARAMS = {
    "model": "medium",
    "correctness_decision": "You are evaluating a student's answer to the question '{{context}}'. The correct answer is '{{answer}}'. Output only 'True' if the student's answer is correct, or 'False' if it is incorrect.",
    "feedback_guidance": "Provide one sentence of feedback explaining why the student's answer was right or wrong.",
    "context": "",
}

def dev():
    """Run the evaluation function from the command line for development purposes.

    Usage: python -m evaluation_function.dev <response> <answer> [--model MODEL] [--params JSON]
    """
    parser = argparse.ArgumentParser(description="Test the evaluation function locally.")
    parser.add_argument("response", help="The student's response")
    parser.add_argument("answer", help="The correct answer")
    parser.add_argument("--model", default=None, help="OpenRouter model ID or tier: small, medium, large, reasoning (overrides params)")
    parser.add_argument("--params", default=None, help="JSON string of params (merged over defaults)")
    args = parser.parse_args()

    params = dict(DEFAULT_PARAMS)
    if args.params:
        params.update(json.loads(args.params))
    if args.model:
        params["model"] = args.model

    result = evaluation_function(args.response, args.answer, params)
    print(result.to_dict())

if __name__ == "__main__":
    dev()
