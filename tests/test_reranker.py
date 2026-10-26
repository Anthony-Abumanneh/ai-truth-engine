"""Step 8.5: evidence reranker (skips when the model isn't downloaded)."""
import math

import pytest

from truth_engine.pipeline.reranker import Reranker, model_dir


@pytest.fixture(scope="module")
def rr():
    if not (model_dir("msmarco") / "model.safetensors").exists():
        pytest.skip("reranker model not downloaded")
    return Reranker("msmarco")


def test_scores_are_finite_and_ranked(rr):
    claim = "A high microerythrocyte count raises vulnerability to severe anemia in alpha-thalassemia."
    on, off = rr.score([(claim, "The increased erythrocyte count and microcytosis in children homozygous for "
                                "alpha(+)-thalassaemia may protect them against severe malarial anaemia."),
                        (claim, "The weather in Paris was sunny on Tuesday.")])
    assert math.isfinite(on) and math.isfinite(off)          # mmap'd weights once gave NaN on this Mac
    assert on > 0.5 > off
