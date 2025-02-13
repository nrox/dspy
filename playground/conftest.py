import json
import os
import re
from datetime import datetime
from pathlib import Path

import pytest
from blib2to3.pgen2.driver import contextmanager
from pydantic import BaseModel

import dspy
from dspy.clients.base_lm import GLOBAL_HISTORY

trace = []

MODEL = [
    "ollama_chat/deepseek-r1:8b-llama-distill-q8_0",
    "ollama_chat/llama3.1:8b-instruct-q8_0",
    "ollama_chat/deepscaler:1.5b-preview-q8_0",
    "ollama_chat/smollm2:1.7b-instruct-q8_0",
    "ollama_chat/llama3.2:3b-instruct-q8_0"
][-1]


def get_lm(model: str = MODEL, api_key: str = "PROVIDER_API_KEY", api_base: str = "http://localhost:11434"):
    lm = dspy.LM(
        model=model,
        api_key=api_key,
        api_base=api_base,
        max_tokens=8 * 1024,
        num_ctx=8 * 1024,
    )
    print("configure llm ", model, "\n", "=" * 50, "\n" * 2)
    return lm


@contextmanager
def conf_dspy():
    start = re.sub(r"\W+", "_", datetime.now().isoformat())
    folder = Path(__file__).parents[1] / "out" / start
    os.makedirs(folder, exist_ok=True)
    lm = get_lm()
    trace.clear()
    dspy.configure(lm=lm, trace=trace)
    yield
    print()
    try:
        if trace:
            trace[0][0].save(path=str(folder / "save.json"))
    except Exception as ex:
        print(ex)

    with open(folder / "traces.txt", "w", encoding="utf8") as fp:
        for i in trace:
            for j in i:
                fp.write(j.model_dump_json(indent=2) if isinstance(j, BaseModel) else str(j))
                fp.write("\n")
            fp.write("\n")

    with open(folder / "history.json", "w", encoding="utf8") as fp:
        json.dump(lm.history, fp, indent=2, default=lambda v: v.model_dump() if isinstance(v, BaseModel) else str(v))

    with open(folder / "global_history.json", "w", encoding="utf8") as fp:
        json.dump(
            GLOBAL_HISTORY, fp, indent=2, default=lambda v: v.model_dump() if isinstance(v, BaseModel) else str(v)
        )

    print("\n\n", "=" * 50, "\n" * 2)


@pytest.fixture(scope="function", autouse=True)
def configure_dspy():
    with conf_dspy():
        yield
