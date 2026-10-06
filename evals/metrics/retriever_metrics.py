from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric

from evals.judge import judge

THRESHOLD = 0.7


def get_retriever_metrics():
    kw = dict(threshold=THRESHOLD, model=judge, include_reason=False)
    return [
        ContextualPrecisionMetric(**kw),
        ContextualRecallMetric(**kw),
    ]