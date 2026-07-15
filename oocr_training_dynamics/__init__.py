"""OOCR training-dynamics and activation-patching research package."""

from oocr_training_dynamics.contracts import CHECKPOINT_STEPS, TrainingCondition
from oocr_training_dynamics.models import MODEL_SPECS, ModelKey

__all__ = ["CHECKPOINT_STEPS", "MODEL_SPECS", "ModelKey", "TrainingCondition"]
