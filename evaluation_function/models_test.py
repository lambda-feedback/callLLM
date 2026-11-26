import json
import os
import unittest
import urllib.request

from .models import DEFAULT_TIER, MODEL_TIERS, PROVIDER_TIERS, get_tier, resolve_models

MODEL_ID = r"^[\w.-]+/[\w.:-]+$"


class TestResolveModels(unittest.TestCase):

    def test_explicit_model_passes_through(self):
        self.assertEqual(resolve_models("openai/gpt-4o-mini"), ["openai/gpt-4o-mini"])

    def test_tier_expands_to_its_models(self):
        self.assertEqual(resolve_models("small"), MODEL_TIERS["small"]["models"])

    def test_resolving_does_not_mutate_tier(self):
        resolve_models("large", ["meta-llama/llama-3.3-70b-instruct"])
        self.assertNotIn("meta-llama/llama-3.3-70b-instruct", MODEL_TIERS["large"]["models"])
        resolve_models("large", ["meta-llama/llama-3.3-70b-instruct"], "google")
        self.assertNotIn(
            "meta-llama/llama-3.3-70b-instruct", PROVIDER_TIERS["google"]["large"]["models"]
        )

    def test_user_fallbacks_follow_tier_models(self):
        self.assertEqual(
            resolve_models("large", ["meta-llama/llama-3.3-70b-instruct"]),
            [*MODEL_TIERS["large"]["models"], "meta-llama/llama-3.3-70b-instruct"],
        )

    def test_invalid_fallbacks_ignored(self):
        self.assertEqual(resolve_models("openai/gpt-4o-mini", "x/y"), ["openai/gpt-4o-mini"])
        self.assertEqual(resolve_models("openai/gpt-4o-mini", ["x/y", 1]), ["openai/gpt-4o-mini"])

    def test_provider_tier_expands_to_providers_models(self):
        self.assertEqual(
            resolve_models("medium", provider="openai"),
            PROVIDER_TIERS["openai"]["medium"]["models"],
        )

    def test_user_fallbacks_follow_provider_tier_models(self):
        self.assertEqual(
            resolve_models("small", ["x/backup"], "deepseek"),
            [*PROVIDER_TIERS["deepseek"]["small"]["models"], "x/backup"],
        )

    def test_provider_ignored_for_explicit_model(self):
        self.assertEqual(
            resolve_models("openai/gpt-4o-mini", provider="google"), ["openai/gpt-4o-mini"]
        )
        self.assertIsNone(get_tier("openai/gpt-4o-mini", "acme"))

    def test_unknown_provider_raises(self):
        with self.assertRaisesRegex(ValueError, "unknown provider"):
            resolve_models("medium", provider="acme")
        with self.assertRaises(ValueError):
            get_tier("medium", "OpenAI")

    def test_get_tier_without_provider_uses_generic_tiers(self):
        self.assertIs(get_tier("reasoning"), MODEL_TIERS["reasoning"])

    def test_tiers_are_well_formed(self):
        self.assertIn(DEFAULT_TIER, MODEL_TIERS)
        for name, tier in MODEL_TIERS.items():
            with self.subTest(tier=name):
                self.assertEqual(name, name.lower())
                self.assertTrue(tier["models"])
                for model in tier["models"]:
                    self.assertRegex(model, MODEL_ID)
                    self.assertNotIn(model, MODEL_TIERS)

    def test_provider_tiers_are_well_formed(self):
        for provider, tiers in PROVIDER_TIERS.items():
            with self.subTest(provider=provider):
                self.assertEqual(provider, provider.lower())
                self.assertEqual(set(tiers), set(MODEL_TIERS))
                for name, tier in tiers.items():
                    self.assertTrue(tier["models"])
                    for model in tier["models"]:
                        self.assertRegex(model, MODEL_ID)
                        # Provider tiers never fall back to another provider.
                        self.assertTrue(model.startswith(f"{provider}/"), (name, model))


def _all_tier_models():
    for name, tier in MODEL_TIERS.items():
        for model in tier["models"]:
            yield name, model
    for provider, tiers in PROVIDER_TIERS.items():
        for name, tier in tiers.items():
            for model in tier["models"]:
                yield f"{provider}:{name}", model


@unittest.skipUnless(
    os.environ.get("CHECK_MODEL_TIERS_ZDR"),
    "set CHECK_MODEL_TIERS_ZDR=1 to check tiers against OpenRouter's live ZDR list",
)
class TestModelTiersZdr(unittest.TestCase):

    def test_every_tier_model_has_zdr_json_endpoint(self):
        with urllib.request.urlopen("https://openrouter.ai/api/v1/endpoints/zdr", timeout=30) as r:
            endpoints = json.load(r)["data"]
        json_capable = {
            e["model_id"] for e in endpoints
            if "response_format" in (e.get("supported_parameters") or [])
        }
        for name, model in _all_tier_models():
            with self.subTest(tier=name, model=model):
                self.assertIn(model, json_capable)
