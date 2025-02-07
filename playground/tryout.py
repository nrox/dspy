import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import pytest

import dspy

trace = []


@pytest.fixture(scope='function', autouse=True)
def configure_dspy(
        model: str = 'ollama/llama3.1:8b-instruct-q8_0',
        api_key: str = "PROVIDER_API_KEY",
        api_base: str = 'http://localhost:11434'):
    lm = dspy.LM(
        model=model,
        api_key=api_key,
        api_base=api_base,
    )
    trace.clear()
    dspy.configure(lm=lm, trace=trace)
    yield
    print()
    try:
        trace[0][0].save(path=str(Path(__file__).parent / 'out' / f'log_{datetime.now().timestamp()}.json'))
    except Exception as ex:
        print(ex)

    for i in trace:
        for j in i:
            print(j)

    print(json.dumps(lm.history, indent=2, default=str))

def test_math():
    math = dspy.ChainOfThought("question -> answer: float")
    math(question="Two dice are tossed. What is the probability that the sum equals two?")


def test_rag():
    def search_wikipedia(query: str) -> list[str]:
        results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
        return [x['text'] for x in results]

    rag = dspy.ChainOfThought('context, question -> response')
    question = "What's the name of the castle that David Gregory inherited?"
    rag(context=search_wikipedia(question), question=question)


def test_classification():
    class Classify(dspy.Signature):
        """Classify sentiment of a given sentence."""

        sentence: str = dspy.InputField()
        sentiment: Literal['positive', 'negative', 'neutral'] = dspy.OutputField()
        confidence: float = dspy.OutputField()

    classify = dspy.Predict(Classify)
    classify(sentence="This book was super fun to read, though not the last chapter.")


def test_extraction():
    class ExtractInfo(dspy.Signature):
        """Extract structured information from text."""

        text: str = dspy.InputField()
        title: str = dspy.OutputField()
        headings: list[str] = dspy.OutputField()
        entities: list[dict[str, str]] = dspy.OutputField(desc="a list of entities and their metadata")

    module = dspy.Predict(ExtractInfo)

    text = "Apple Inc. announced its latest iPhone 14 today." \
           "The CEO, Tim Cook, highlighted its new features in a press release."
    response = module(text=text)

    print(response.title)
    print(response.headings)
    print(response.entities)
