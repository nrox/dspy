import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

import pytest
from blib2to3.pgen2.driver import contextmanager
from pydantic import BaseModel

import dspy
from dspy.datasets import HotPotQA

trace = []

MODEL = [
    "ollama_chat/deepseek-r1:8b-llama-distill-q8_0",
    'ollama_chat/llama3.1:8b-instruct-q8_0'
][1]


def get_lm(
        model: str = MODEL,
        api_key: str = "PROVIDER_API_KEY",
        api_base: str = 'http://localhost:11434'):
    lm = dspy.LM(
        model=model,
        api_key=api_key,
        api_base=api_base,
        max_tokens=8 * 1024,
        num_ctx=8 * 1024,
    )
    print("configure llm ", model)
    return lm


@contextmanager
def conf_dspy():
    start = re.sub(r'\W+', '_', datetime.now().isoformat())
    folder = Path(__file__).parents[1] / 'out' / start
    os.makedirs(folder, exist_ok=True)
    lm = get_lm()
    trace.clear()
    dspy.configure(lm=lm, trace=trace)
    yield
    print()
    try:
        trace[0][0].save(path=str(folder / 'save.json'))
    except Exception as ex:
        print(ex)

    with open(folder / 'traces.txt', 'w', encoding='utf8') as fp:
        for i in trace:
            for j in i:
                fp.write(j.model_dump_json(indent=2) if isinstance(j, BaseModel) else str(j))
                fp.write("\n")
            fp.write("\n")

    with open(folder / 'history.json', 'w', encoding='utf8') as fp:
        json.dump(lm.history, fp, indent=2, default=lambda v: v.model_dump() if isinstance(v, BaseModel) else str(v))


@pytest.fixture(scope='function', autouse=True)
def configure_dspy():
    with conf_dspy():
        yield


def test_math():
    math = dspy.ChainOfThought("question -> answer: float")
    math(question="Two dice are tossed. What is the probability that the sum equals two?")


def test_rag():
    def search_wikipedia(query: str) -> list[str]:
        results: list[dict] = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
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
    classify(sentence="This book was super fun to read. The best book ever.")


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


def test_agents():
    def evaluate_math(expression: str):
        return dspy.PythonInterpreter({}).execute(expression)

    def search_wikipedia(query: str):
        results: list[dict] = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
        return [x['text'] for x in results]

    react = dspy.ReAct("question -> answer: float", tools=[evaluate_math, search_wikipedia])

    pred = react(question="What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle?")
    print(pred.answer)


def test_multistage_pipeline():
    class Outline(dspy.Signature):
        """Outline a thorough overview of a topic."""

        topic: str = dspy.InputField()
        title: str = dspy.OutputField()
        sections: list[str] = dspy.OutputField()
        section_subheadings: dict[str, list[str]] = dspy.OutputField(desc="mapping from section headings to subheadings")

    class DraftSection(dspy.Signature):
        """Draft a top-level section of an article."""

        topic: str = dspy.InputField()
        section_heading: str = dspy.InputField()
        section_subheadings: list[str] = dspy.InputField()
        content: str = dspy.OutputField(desc="markdown-formatted section")

    class DraftArticle(dspy.Module):
        def __init__(self):
            self.build_outline = dspy.ChainOfThought(Outline)
            self.draft_section = dspy.ChainOfThought(DraftSection)

        def forward(self, topic):
            outline = self.build_outline(topic=topic)
            sections = []
            for heading, subheadings in outline.section_subheadings.items():
                section, subheadings = f"## {heading}", [f"### {subheading}" for subheading in subheadings]
                section = self.draft_section(topic=outline.title, section_heading=section, section_subheadings=subheadings)
                sections.append(section.content)
            return dspy.Prediction(title=outline.title, sections=sections)

    draft_article = DraftArticle()
    article = draft_article(topic="World Cup 2022")
    print(article)


def test_optimize_react():
    def search_wikipedia(query: str) -> list[str]:
        results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
        return [x['text'] for x in results]

    trainset = [x.with_inputs('question') for x in HotPotQA(train_seed=2024, train_size=500).train]
    react = dspy.ReAct("question -> answer", tools=[search_wikipedia])

    tp = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", num_threads=24)
    optimized_react = tp.compile(react, trainset=trainset)


if __name__ == "__main__":
    print("started")
    with conf_dspy():
        test_multistage_pipeline()
