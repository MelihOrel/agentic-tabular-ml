"""atml: agentic tabular ML platform.

Layers (each importable on its own, no Streamlit dependency):
    atml.io          -> encoding/delimiter aware loading of CSV and Excel
    atml.validation  -> schema guard, ML readiness guard, AST-safe formulas
    atml.processing  -> cleaning operations with an undo/redo history
    atml.modeling    -> task detection, cross-validated AutoML, persistence
    atml.stats       -> hypothesis tests, clustering, PCA, anomaly detection
    atml.agent       -> LangChain tools wrapping the layers above + ReAct agent
"""

from atml.config import load_config

__all__ = ["load_config"]
__version__ = "0.1.0"
