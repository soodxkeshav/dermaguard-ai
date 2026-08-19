"""Load and cache the DermaGuard PyTorch classifier."""

import os
from pathlib import Path
from typing import Any

import torch
from torchvision import models

try:
    import streamlit as st
except ModuleNotFoundError:
    class _StreamlitFallback:
        @staticmethod
        def cache_resource(**_kwargs):
            return lambda function: function

    st = _StreamlitFallback()


CLASS_NAMES = ("benign", "malignant", "non-neoplastic")
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "resnet18_best.pth"


def _state_dict_from_checkpoint(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                checkpoint = candidate
                break
        if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint
    raise ValueError(
        "The checkpoint must contain a PyTorch state dictionary under "
        "'model_state_dict', 'state_dict', or 'model'."
    )


def build_model(state_dict: dict[str, torch.Tensor] | None = None) -> torch.nn.Module:
    model = models.resnet18(weights=None)
    if state_dict is not None and "fc.1.weight" in state_dict:
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES)),
        )
    else:
        model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


@st.cache_resource(show_spinner=False)
def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> tuple[torch.nn.Module, torch.device]:
    """Load the supplied checkpoint once per Streamlit process."""
    path = Path(model_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found at '{path}'. Add the supplied file there "
            "or set the MODEL_PATH environment variable."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    state_dict = _state_dict_from_checkpoint(checkpoint)
    model = build_model(state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model, device
