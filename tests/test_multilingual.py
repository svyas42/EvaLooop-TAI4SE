"""Tests for the multilingual summarization loop types and their config wiring."""

import importlib
import logging
import sys
import types

import pytest

from evaloop.core.config import STANDARD_LOOP_TYPE, SUPPORTED_LOOP_TYPES, EvaluationConfig
from evaloop.tasks import TaskFactory
from evaloop.tasks.multilingual_prompts import (
    MULTILINGUAL_SUMMARIZATION_TEMPLATES,
    SUMMARY_OPENINGS,
)

from conftest import ASSERT_LINE, GENERATION_CONFIG, MBPP_STYLE_PROMPT, SUMMARIZATION_CONFIG

LANGUAGES = {"summarization_hindi": ("hindi", "Hindi"), "summarization_spanish": ("spanish", "Spanish")}


def make_config(tmp_path, loop_type):
    return EvaluationConfig(model="test-model", loop_type=loop_type, output_dir=str(tmp_path))


class TestMultilingualTemplates:
    @pytest.mark.parametrize("loop_type", sorted(MULTILINGUAL_SUMMARIZATION_TEMPLATES))
    def test_differs_from_standard_only_in_language(self, loop_type):
        # Undoing the two language changes must give back the standard prompt exactly
        key, name = LANGUAGES[loop_type]
        template = MULTILINGUAL_SUMMARIZATION_TEMPLATES[loop_type]
        restored = template.replace(f" in {name}", "").replace(SUMMARY_OPENINGS[key], "write a python function to")
        assert restored == SUMMARIZATION_CONFIG["prompt_template"]

    @pytest.mark.parametrize("loop_type", sorted(MULTILINGUAL_SUMMARIZATION_TEMPLATES))
    def test_prompt_embeds_code_and_opening(self, loop_type):
        key, _ = LANGUAGES[loop_type]
        task = TaskFactory().create_task(
            {"type": "code_summarization", "prompt_template": MULTILINGUAL_SUMMARIZATION_TEMPLATES[loop_type]}
        )
        prompt = task.create_prompt("def f(x): return x")
        assert "```\ndef f(x): return x\n```" in prompt
        assert prompt.endswith(f"```\n{SUMMARY_OPENINGS[key]}\n```")

    def test_hindi_opening_is_devanagari(self):
        letters = [c for c in SUMMARY_OPENINGS["hindi"] if c.isalpha()]
        assert letters and all("ऀ" <= c <= "ॿ" for c in letters)

    def test_postprocess_retains_assert_line_for_non_english_summary(self):
        task = TaskFactory().create_task(
            {"type": "code_summarization", "prompt_template": MULTILINGUAL_SUMMARIZATION_TEMPLATES["summarization_hindi"]}
        )
        summary = "एक पायथन फ़ंक्शन लिखें जो दो सूचियों के साझा तत्व ढूंढे"
        result = task.postprocess(summary, {"cycle_input": MBPP_STYLE_PROMPT, "task_id": "Mbpp/2", "cycle_number": 1})
        assert result == '"""\n' + summary + "\n" + ASSERT_LINE + '\n"""\n'


class TestLoopTypeConfig:
    def test_default_is_standard_loop(self, tmp_path):
        config = EvaluationConfig(model="test-model", output_dir=str(tmp_path))
        assert config.loop_type == STANDARD_LOOP_TYPE
        assert config.get_task_configs(config.loop_type) == [GENERATION_CONFIG, SUMMARIZATION_CONFIG]

    @pytest.mark.parametrize("loop_type", sorted(MULTILINGUAL_SUMMARIZATION_TEMPLATES))
    def test_multilingual_loop_changes_only_summarization_prompt(self, tmp_path, loop_type):
        config = make_config(tmp_path, loop_type)
        generation_config, summarization_config = config.get_task_configs(config.loop_type)
        assert generation_config == GENERATION_CONFIG
        assert summarization_config == {
            "type": "code_summarization",
            "prompt_template": MULTILINGUAL_SUMMARIZATION_TEMPLATES[loop_type],
        }

    def test_multilingual_loop_does_not_alter_standard_configs(self, tmp_path):
        config = make_config(tmp_path, "summarization_spanish")
        config.get_task_configs(config.loop_type)
        assert config.get_task_configs(STANDARD_LOOP_TYPE) == [GENERATION_CONFIG, SUMMARIZATION_CONFIG]

    def test_unknown_loop_type_raises(self, tmp_path):
        with pytest.raises(ValueError, match="loop_type"):
            make_config(tmp_path, "summarization_klingon")

    def test_loop_type_in_dict(self, tmp_path):
        assert make_config(tmp_path, "summarization_hindi").to_dict()["loop_type"] == "summarization_hindi"

    def test_supported_loop_types(self):
        assert SUPPORTED_LOOP_TYPES == [STANDARD_LOOP_TYPE, "summarization_hindi", "summarization_spanish"]


@pytest.fixture
def evaluator_cls(monkeypatch):
    """EvaLoopEvaluator imported with its torch/vLLM/EvalPlus dependencies stubbed out."""
    stubs = {
        "evaloop.models.factory": {"ModelFactory": object},
        "evaloop.evaluation.py_testing": {"EvalPlusCodeTester": object, "load_mbpp_dataset": None},
    }
    for name, attributes in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    # Force a fresh import against the stubs; monkeypatch restores the original entry afterwards
    monkeypatch.setitem(sys.modules, "evaloop.core.evaluator", None)
    del sys.modules["evaloop.core.evaluator"]
    return importlib.import_module("evaloop.core.evaluator").EvaLoopEvaluator


def test_tester_error_is_recorded_instead_of_crashing(evaluator_cls):
    class RaisingTester:
        def test_code(self, code, task_id, entry_point):
            raise RuntimeError("sandbox failure")

    evaluator = evaluator_cls.__new__(evaluator_cls)
    evaluator.logger = logging.getLogger(__name__)
    factory = TaskFactory()
    tasks = [factory.create_task(GENERATION_CONFIG), factory.create_task(SUMMARIZATION_CONFIG)]
    cycle_result = {"task_id": "Mbpp/2", "task_1_output": "def similar_elements(a, b): pass"}

    success = evaluator._test_and_validate_output(
        cycle_result, tasks, "Mbpp/2", {"entry_point": "similar_elements"}, RaisingTester()
    )

    assert success is False
    assert cycle_result["test_results"] == {"success": False, "error": "sandbox failure"}
