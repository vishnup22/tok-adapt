"""tok_adapt: adapt, extend, prune, and initialize HF tokenizers & embeddings.

Public API re-exports the main classes so callers can do:

    from tok_adapt import VocabularyExpander, EmbeddingInitializer, VocabularyPruner, FertilityEvaluator
"""

from tok_adapt.expansion import VocabularyExpander
from tok_adapt.initialization import EmbeddingInitializer
from tok_adapt.metrics import FertilityEvaluator
from tok_adapt.pruning import VocabularyPruner

__version__ = "0.1.0"

__all__ = [
    "VocabularyExpander",
    "EmbeddingInitializer",
    "VocabularyPruner",
    "FertilityEvaluator",
    "__version__",
]
