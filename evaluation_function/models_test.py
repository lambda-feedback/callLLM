import json
import os
import unittest
import urllib.request

from .models import DEFAULT_TIER, MODEL_TIERS, resolve_models


class TestResolveModels(unittest.TestCase):

    def test_explicit_model_passes_through(self):
        self.assertEqual(resolve_models("openai/gpt-4o-mini"), ["openai/gpt-4o-mini"])

    def test_tier_expands_to_its_models(self):
        self.assertEqual(resolve_models("small"), MODEL_TIERS["small"]["models"])

    def test_resolving_does_not_mutate_tier(self):
        resolve_models("large", ["meta-llama/llama-3.3-70b-instruct"])
        self.assertNotIn("meta-llama/llama-3.3-70b-instruct", MODEL_TIERS["large"]["models"])

    def test_user_fallbacks_follow_tier_models(self):
        self.assertEqual(
            resolve_models("large", ["meta-llama/llama-3.3-70b-instruct"]),
            [*MODEL_TIERS["large"]["models"], "meta-llama/llama-3.3-70b-instruct"],
        )

    def test_invalid_fallbacks_ignored(self):
        self.assertEqual(resolve_models("openai/gpt-4o-mini", "x/y"), ["openai/gpt-4o-mini"])
        self.assertEqual(resolve_models("openai/gpt-4o-mini", ["x/y", 1]), ["openai/gpt-4o-mini"])

    def test_tiers_are_well_formed(self):
        self.assertIn(DEFAULT_TIER, MODEL_TIERS)
        for name, tier in MODEL_TIERS.items():
            with self.subTest(tier=name):
                self.assertEqual(name, name.lower())
                self.assertTrue(tier["models"])
                for model in tier["models"]:
                    self.assertRegex(model, r"^[\w.-]+/[\w.:-]+$")
                    self.assertNotIn(model, MODEL_TIERS)


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
        for name, tier in MODEL_TIERS.items():
            for model in tier["models"]:
                with self.subTest(tier=name, model=model):
                    self.assertIn(model, json_capable)
