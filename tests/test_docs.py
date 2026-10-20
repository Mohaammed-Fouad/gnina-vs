"""Documentation / packaging consistency (items 1, 16, 18)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
PYPROJECT = (ROOT / "pyproject.toml").read_text()


def test_readme_uses_valid_cnn_scoring_values():
    assert not re.search(r"--cnn-scoring\s+(refine|full)\b", README)
    assert "refinement" in README


def test_readme_does_not_claim_polar_hydrogens_only():
    assert "polar hydrogens" not in README.lower() or "not" in README.lower().split("polar hydrogens")[0][-80:]
    assert "gasteiger" not in README.lower()


def test_scipy_consistency_between_readme_and_pyproject():
    deps = re.search(r"^dependencies\s*=\s*\[(.*?)\]", PYPROJECT, re.S | re.M).group(1)
    required = "scipy" in deps.lower()
    readme_says_optional = re.search(r"scipy[^\n]{0,80}optional|optional[^\n]{0,80}scipy", README, re.I)
    assert not (required and readme_says_optional), "scipy is required in pyproject but 'optional' in README"
