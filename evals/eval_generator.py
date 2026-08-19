import os
import json

from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from deepeval.models import LocalModel
from deepeval import evaluate
from deepeval.evaluate.configs import CacheConfig

from dotenv import load_dotenv

from src.generator import generate

load_dotenv()

# deepeval has no built-in Groq wrapper, but Groq exposes an OpenAI-compatible
# endpoint, so we point the generic LocalModel at it using GROQ_API_KEY.
JUDGE_MODEL = LocalModel(
    model="openai/gpt-oss-120b",
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

path = 'goldendata/faithfulness_data.json'

with open(path) as f:
    data = json.load(f)

test_cases = []

for i in data:
    test_cases.append(LLMTestCase(
            input=i['query'],
            retrieval_context=i['ideal_context'],
            actual_output=generate(i['query'], i['ideal_context'])
        ))

metrics = [FaithfulnessMetric(model=JUDGE_MODEL, threshold=0.6)]

evaluate(
    test_cases=test_cases,
    metrics=metrics,
    cache_config=CacheConfig(write_cache=False, use_cache=False),
)
