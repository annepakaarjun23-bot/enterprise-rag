from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCaseParams

from evals.judge import judge

THRESHOLD = 0.7

CORRECTNESS_CRITERIA = (
    "Judge whether the actual output conveys the same facts and instructions as "
    "the expected output. Penalize contradictions and missing key points. "
    "Do not penalize extra detail or different wording."
)


def get_generation_metrics():
    return [
        FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=False),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=False),
        # GEval(
        #     name="Correctness",
        #     criteria=CORRECTNESS_CRITERIA,
        #     evaluation_params=[
        #         LLMTestCaseParams.INPUT,
        #         LLMTestCaseParams.ACTUAL_OUTPUT,
        #         LLMTestCaseParams.EXPECTED_OUTPUT,
        #     ],
        #     threshold=THRESHOLD,
        #     model=judge,
        # ),
    ]