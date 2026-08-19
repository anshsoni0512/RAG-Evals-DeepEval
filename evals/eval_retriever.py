import json
import os
import sys
from pathlib import Path

# make sure the project root (and therefore `src`) is importable regardless
# of whether this script is run directly or as a module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models import LocalModel
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig

from src.retriver import build_retriever

load_dotenv()

GOLDEN_PATH = "goldendata/golden_retriever.json"
THRESHOLD = 0.7

# deepeval has no built-in Groq wrapper, but Groq exposes an OpenAI-compatible
# endpoint, so we point the generic LocalModel at it using GROQ_API_KEY.
JUDGE_MODEL = LocalModel(
    model="openai/gpt-oss-120b",
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

# first load the golden dataset

with open(GOLDEN_PATH) as f:
    data = json.load(f)


retriever = build_retriever()

test_cases = []

for g in data:
    retrived_answer = retriever.invoke(g["query"])
    retrieval_context = [doc.page_content for doc in retrived_answer]

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            expected_output=g["ideal_answer"],
            retrieval_context=retrieval_context,
        )
    )

metrics = [
    ContextualRecallMetric(threshold=THRESHOLD, include_reason=True, model=JUDGE_MODEL),
    ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
]

evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(max_concurrent=1, throttle_value=16),
    cache_config=CacheConfig(write_cache=False, use_cache=False),
)