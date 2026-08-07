import unittest


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str) -> unittest.TestSuite:
    suite = unittest.TestSuite()
    for module_name in (
        "tests.test_prompt_contract_chart_scenarios",
        "tests.test_prompt_contract_statistical_scenarios",
    ):
        suite.addTests(loader.loadTestsFromName(module_name))
    return suite


if __name__ == "__main__":
    unittest.main()