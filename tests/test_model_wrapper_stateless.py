import copy
import importlib.util
import os
import sys
import types
from types import SimpleNamespace

import pytest

# Only stub openai when it isn't installed; the client is replaced below either way
if "openai" not in sys.modules:
    try:
        import openai  # noqa: F401
    except ImportError:
        stub = types.ModuleType("openai")
        stub.OpenAI = lambda **kwargs: None
        sys.modules["openai"] = stub

# Load the module file directly: shared/__init__ pulls in the heavy training stack
_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shared", "model_wrapper.py")
_spec = importlib.util.spec_from_file_location("model_wrapper", _path)
model_wrapper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(model_wrapper)
STATELESS_NOTICE, ModelWrapper = model_wrapper.STATELESS_NOTICE, model_wrapper.ModelWrapper


class FakeCompletions:
    def __init__(self):
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs["messages"]))
        reply = f"<answer>{len(self.requests)}</answer>"
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=reply, reasoning=None),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )


def _wrapper(stateless):
    model = ModelWrapper("test-model", "vllm", stateless=stateless)
    completions = FakeCompletions()
    model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    model.init_chat("system prompt")
    return model, completions


@pytest.mark.parametrize("stateless", [False, True])
def test_requests_and_transcript(stateless):
    model, completions = _wrapper(stateless)
    for message in ("turn 1", "turn 2", "turn 3"):
        model.send_message(message)

    sent_lengths = [len(r) for r in completions.requests]
    expected_transcript = [
        "turn 1", "<answer>1</answer>",
        "turn 2", "<answer>2</answer>",
        "turn 3", "<answer>3</answer>",
    ]
    assert [m["content"] for m in model.full_history[1:]] == expected_transcript

    if stateless:
        assert sent_lengths == [2, 2, 2]
        assert completions.requests[2][1]["content"] == "turn 3"
        assert model.full_history[0]["content"] == "system prompt" + STATELESS_NOTICE
    else:
        assert sent_lengths == [2, 4, 6]
        assert model.full_history == model.history
        assert model.history[0]["content"] == "system prompt"


def test_init_chat_clears_dropped_history():
    model, _ = _wrapper(stateless=True)
    model.send_message("turn 1")
    model.send_message("turn 2")
    model.init_chat("new system")
    assert model.full_history == [{"role": "system", "content": "new system" + STATELESS_NOTICE}]
