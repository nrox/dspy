import random

import ollama
import ujson

import dspy
from dspy.evaluate import SemanticF1
from dspy.utils import download


def test_basic_predict():
    qa = dspy.Predict("question: str -> response: str")
    response = qa(question="what are high memory and low memory on linux?")
    print(response.response)


def test_basic_cot():
    cot = dspy.ChainOfThought("question -> response")
    response = cot(question="should curly braces appear on their own line?")
    print(response.response)


def test_examples():
    # Download question--answer pairs from the RAG-QA Arena "Tech" dataset.
    download("https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_examples.jsonl")

    with open("ragqa_arena_tech_examples.jsonl") as f:
        data = [ujson.loads(line) for line in f]

    data = [dspy.Example(**d).with_inputs("question") for d in data]

    # Let's pick an `example` here from the data.
    example = data[2]
    print(example)

    random.Random(0).shuffle(data)
    scale = 10
    trainset, devset, testset = data[:2*scale], data[2*scale:5*scale], data[5*scale:10*scale]

    # Instantiate the metric.
    metric = SemanticF1(decompositional=True)

    # Produce a prediction from our `cot` module, using the `example` above as input.
    cot = dspy.ChainOfThought("question -> response")
    pred = cot(**example.inputs())

    # Compute the metric score for the prediction.
    score = metric(example, pred)

    print(f"Question: \t {example.question}\n")
    print(f"Gold Response: \t {example.response}\n")
    print(f"Predicted Response: \t {pred.response}\n")
    print(f"Semantic F1 Score: {score:.2f}")

    if False:
        return

    # Define an evaluator that we can re-use.
    evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=24,
                            display_progress=True, display_table=2)

    # Evaluate the Chain-of-Thought program.
    evaluate(cot)


def get_evaluator(scale = 10) -> dspy.Evaluate:
    # Download question--answer pairs from the RAG-QA Arena "Tech" dataset.
    download("https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_examples.jsonl")

    with open("ragqa_arena_tech_examples.jsonl") as f:
        data = [ujson.loads(line) for line in f]

    data = [dspy.Example(**d).with_inputs("question") for d in data]


    random.Random(0).shuffle(data)
    
    trainset, devset, testset = data[:2*scale], data[2*scale:5*scale], data[5*scale:10*scale]

    # Instantiate the metric.
    metric = SemanticF1(decompositional=True)

    evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=24,
                            display_progress=True, display_table=2)

    return evaluate


def ollama_embed_fn(text: str):
    # "nomic-embed-text:latest"
    embed_model = "nomic-embed-text:latest"
    return ollama.embed(model=embed_model, input=text).embeddings


def test_basic_rag():
    class RAG(dspy.Module):
        def __init__(self, search: dspy.retrievers.Embeddings):
            self.respond = dspy.ChainOfThought('context, question -> response')
            self.search = search

        def forward(self, question):
            context = self.search(question).passages
            return self.respond(context=context, question=question)
    
    dataset_name = "ragqa_arena_tech_corpus.jsonl"
    download(f"https://huggingface.co/dspy/cache/resolve/main/{dataset_name}")

    max_characters = 6000  # for truncating >99th percentile of documents
    topk_docs_to_retrieve = 5  # number of documents to retrieve per search query

    with open("ragqa_arena_tech_corpus.jsonl") as f:
        corpus = [ujson.loads(line)['text'][:max_characters] for line in f][:500]
        print(f"Loaded {len(corpus)} documents. Will encode them below.")

    embedder = dspy.Embedder(ollama_embed_fn, batch_size=1)
    search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=topk_docs_to_retrieve)
    rag = RAG(search=search)
    response = rag(question="what are high memory and low memory on linux?")
    print(response)
    evaluation = get_evaluator()(RAG(search=search))
    print(evaluation)
