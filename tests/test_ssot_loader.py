import unittest
from typing import Any, Dict, List
from unittest import mock

from src.shared import ssot_loader


class SsotLoaderTests(unittest.TestCase):
    def test_metric_text_lookup_supports_localized_synonyms_and_descriptions(
        self,
    ) -> None:
        items: List[Dict[str, Any]] = [
            {
                "canonical": "THROMBECTOMY",
                "synonyms": {
                    "en": ["clot removal", "mechanical thrombectomy"],
                    "el": ["αφαίρεση θρόμβου"],
                },
                "description": {
                    "en": "Description for THROMBECTOMY",
                    "el": "Περιγραφή για ΘΡΟΜΒΕΚΤΟΜΙΑ",
                },
                "data_type": "Enum",
            }
        ]

        ssot_loader.get_metric_text_lookup.cache_clear()
        with mock.patch.object(ssot_loader, "_load_yaml", return_value=items):
            lookup = ssot_loader.get_metric_text_lookup()

        self.assertEqual(lookup["clot removal"]["canonical"], "THROMBECTOMY")
        self.assertEqual(lookup["αφαίρεση θρόμβου"]["canonical"], "THROMBECTOMY")
        self.assertEqual(
            lookup["clot removal"]["descriptions"]["el"],
            "Περιγραφή για ΘΡΟΜΒΕΚΤΟΜΙΑ",
        )

    def tearDown(self) -> None:
        ssot_loader.get_metric_text_lookup.cache_clear()


class ResolveCountryCodeTests(unittest.TestCase):
    _ITEMS: List[Dict[str, Any]] = [
        {
            "canonical": "CZ",
            "synonyms": {"en": ["Czechia", "Czech Republic"], "cs": ["Česko"]},
        },
        {
            "canonical": "ES",
            "synonyms": {"en": ["Spain", "Espana", "España"], "cs": ["Španělsko"]},
        },
    ]

    def setUp(self) -> None:
        ssot_loader._canonical_lookup.cache_clear()
        self._patcher = mock.patch.object(ssot_loader, "_load_yaml", return_value=self._ITEMS)
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        ssot_loader._canonical_lookup.cache_clear()

    def test_resolves_bare_iso_code(self) -> None:
        self.assertEqual(ssot_loader.resolve_country_code("cz"), "CZ")

    def test_resolves_english_synonym(self) -> None:
        self.assertEqual(ssot_loader.resolve_country_code("Czech Republic"), "CZ")
        self.assertEqual(ssot_loader.resolve_country_code("Czechia"), "CZ")

    def test_resolves_diacritic_variant(self) -> None:
        self.assertEqual(ssot_loader.resolve_country_code("España"), "ES")
        self.assertEqual(ssot_loader.resolve_country_code("Espana"), "ES")

    def test_unknown_country_returns_none(self) -> None:
        self.assertIsNone(ssot_loader.resolve_country_code("Narnia"))


if __name__ == "__main__":
    unittest.main()
