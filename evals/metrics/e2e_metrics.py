from evals.metrics.generation_metrics import get_generation_metrics
from evals.metrics.retriever_context_relavency_metrics import get_context_relavency_metrics
from evals.metrics.retriever_metrics import get_retriever_metrics


def get_e2e_metrics():
    return [
        *get_retriever_metrics(),
        # *get_context_relavency_metrics(),
        *get_generation_metrics(),
    ]