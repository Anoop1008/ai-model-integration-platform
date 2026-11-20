"""Inference pipeline: preprocess -> infer -> postprocess, each a separate method
so each step can be tested and timed on its own."""

import json
import os
import time
import uuid
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from app.schemas import ChurnRequest, ChurnResponse

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
FEATURES = ["tenure_months", "monthly_charges", "total_charges", "support_tickets", "contract_type"]


def risk_level(p: float) -> str:
    if p < 0.3:
        return "Low"
    if p < 0.6:
        return "Medium"
    return "High"


class InferencePipeline:
    def __init__(self, model: tf.keras.Model, preprocessor, training_stats: dict):
        self.model = model
        self.preprocessor = preprocessor
        self.training_stats = training_stats
        self.model_version = training_stats["model_version"]
        # Compile the forward pass into a TF graph once. Eager Keras calls spend most of
        # their time in Python (holding the GIL); the graph call measured ~6x faster here.
        n_features = len(preprocessor.get_feature_names_out())
        self._forward = tf.function(
            self._model_call, input_signature=[tf.TensorSpec([None, n_features], tf.float32)]
        )

    def _model_call(self, x):
        return self.model(x, training=False)  # training=False disables Dropout

    @classmethod
    def load(cls, artifacts_dir: Path = ARTIFACTS_DIR) -> "InferencePipeline":
        # Loading a Keras model takes seconds; callers do this once at startup.
        model = tf.keras.models.load_model(artifacts_dir / "churn_model.keras")
        preprocessor = joblib.load(artifacts_dir / "preprocessor.joblib")
        stats = json.loads((artifacts_dir / "training_stats.json").read_text())
        return cls(model, preprocessor, stats)

    def preprocess(self, req: ChurnRequest) -> np.ndarray:
        # Same fitted ColumnTransformer as training, so scaling/encoding cannot drift
        # between training and serving.
        row = pd.DataFrame([req.model_dump()], columns=FEATURES)
        return self.preprocessor.transform(row).astype("float32")

    def infer(self, x: np.ndarray) -> float:
        # Calling the model directly is much faster than model.predict() for one row
        # (predict builds a tf.data pipeline per call).
        return float(self._forward(x).numpy()[0, 0])

    def postprocess(self, prob: float, latency_ms: float) -> ChurnResponse:
        return ChurnResponse(
            request_id=uuid.uuid4().hex,
            prediction="Churn" if prob >= 0.5 else "Retain",
            confidence_score=round(max(prob, 1 - prob), 4),
            churn_probability=round(prob, 4),
            risk_level=risk_level(prob),
            inference_latency_ms=round(latency_ms, 3),
            model_version=self.model_version,
        )

    def predict(self, req: ChurnRequest) -> ChurnResponse:
        start = time.perf_counter()
        prob = self.infer(self.preprocess(req))
        return self.postprocess(prob, (time.perf_counter() - start) * 1000)

    def warm_up(self) -> None:
        # The first call traces the tf.function graph and is slow; pay that cost at startup
        # instead of on the first user's request.
        self.predict(
            ChurnRequest(
                tenure_months=12,
                monthly_charges=70.0,
                total_charges=840.0,
                support_tickets=1,
                contract_type="one-year",
            )
        )
