from deepeval.metrics import ContextualRelevancyMetric

from evals.judge import judge

THRESHOLD = 0.7


def get_context_relavency_metrics():
    kw = dict(threshold=THRESHOLD, model=judge, include_reason=False)
    return [
        ContextualRelevancyMetric(**kw)
    ]