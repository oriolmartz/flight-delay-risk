"""PyTorch estimators for leakage-safe tabular flight-delay modelling.

The estimators deliberately expose a scikit-learn compatible ``fit`` /
``predict_proba`` interface so they participate in the exact same temporal
selection, calibration and test protocol as the classical candidates.
Categorical variables are represented with train-fitted embeddings rather than
high-dimensional one-hot vectors.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_SEED

try:  # Optional heavyweight dependency; imported lazily by candidate builders.
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - exercised only without torch installed
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "Neural FlightRisk candidates require PyTorch. Install "
            "requirements-advanced.txt or `pip install torch`."
        ) from _TORCH_IMPORT_ERROR


def _embedding_dim(cardinality: int) -> int:
    """Balanced embedding width for low and high-cardinality categories."""
    return int(min(48, max(4, round(1.6 * max(cardinality, 2) ** 0.56))))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    _require_torch()
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class _EncodedFrame:
    categories: Any
    numerics: Any


class _EmbeddingMLP(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        n_numeric: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, _embedding_dim(cardinality)) for cardinality in cardinalities]
        )
        input_dim = sum(layer.embedding_dim for layer in self.embeddings) + n_numeric
        layers: list[nn.Module] = []
        current = input_dim
        for width in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current, width),
                    nn.LayerNorm(width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current = width
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, categories: Any, numerics: Any) -> Any:
        embedded = [layer(categories[:, index]) for index, layer in enumerate(self.embeddings)]
        inputs = torch.cat([*embedded, numerics], dim=1)
        return self.network(inputs).squeeze(1)


class _FTTransformer(nn.Module):
    """Compact FT-Transformer-style network for mixed tabular features."""

    def __init__(
        self,
        cardinalities: list[int],
        n_numeric: int,
        d_token: int,
        n_heads: int,
        n_layers: int,
        ff_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_token % n_heads != 0:
            raise ValueError("d_token must be divisible by n_heads")
        self.category_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_token) for cardinality in cardinalities]
        )
        self.numeric_weight = nn.Parameter(torch.empty(n_numeric, d_token))
        self.numeric_bias = nn.Parameter(torch.zeros(n_numeric, d_token))
        self.column_embeddings = nn.Parameter(
            torch.empty(1, len(cardinalities) + n_numeric + 1, d_token)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * ff_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, 1))
        nn.init.xavier_uniform_(self.numeric_weight)
        nn.init.normal_(self.column_embeddings, std=0.02)

    def forward(self, categories: Any, numerics: Any) -> Any:
        category_tokens = torch.stack(
            [layer(categories[:, index]) for index, layer in enumerate(self.category_embeddings)],
            dim=1,
        )
        numeric_tokens = numerics.unsqueeze(-1) * self.numeric_weight.unsqueeze(0)
        numeric_tokens = numeric_tokens + self.numeric_bias.unsqueeze(0)
        cls = self.cls_token.expand(categories.shape[0], -1, -1)
        tokens = torch.cat([cls, category_tokens, numeric_tokens], dim=1)
        tokens = tokens + self.column_embeddings
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0]).squeeze(1)


class TorchTabularClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible embedding MLP or FT-Transformer classifier.

    The last ``validation_fraction`` of each already chronological training
    frame is used for early stopping. This inner validation tail is never used
    for public candidate selection or final test reporting.
    """

    def __init__(
        self,
        architecture: str = "mlp",
        hidden_dims: tuple[int, ...] = (192, 96, 48),
        d_token: int = 48,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_multiplier: int = 4,
        dropout: float = 0.15,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 1024,
        epochs: int = 18,
        patience: int = 4,
        validation_fraction: float = 0.12,
        min_delta: float = 1e-4,
        gradient_clip_norm: float = 1.0,
        random_state: int = RANDOM_SEED,
        device: str = "auto",
        verbose: bool = False,
    ) -> None:
        self.architecture = architecture
        self.hidden_dims = hidden_dims
        self.d_token = d_token
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ff_multiplier = ff_multiplier
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.min_delta = min_delta
        self.gradient_clip_norm = gradient_clip_norm
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _resolve_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return self.device

    def _fit_encoders(self, X: pd.DataFrame) -> None:
        self.category_maps_ = {}
        self.category_cardinalities_ = []
        for feature in CATEGORICAL_FEATURES:
            values = X[feature].astype(str).fillna("__MISSING__")
            ordered = sorted(values.unique().tolist())
            mapping = {value: index + 1 for index, value in enumerate(ordered)}
            self.category_maps_[feature] = mapping
            self.category_cardinalities_.append(len(mapping) + 1)

        numeric = X[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce").astype(float)
        means = numeric.mean().fillna(0.0)
        stds = numeric.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        self.numeric_means_ = means.to_dict()
        self.numeric_stds_ = stds.to_dict()

    def _encode(self, X: pd.DataFrame) -> _EncodedFrame:
        missing = [feature for feature in [*CATEGORICAL_FEATURES, *NUMERIC_FEATURES] if feature not in X]
        if missing:
            raise KeyError(f"Missing neural model features: {missing}")
        categories = np.column_stack(
            [
                X[feature]
                .astype(str)
                .fillna("__MISSING__")
                .map(self.category_maps_[feature])
                .fillna(0)
                .astype(np.int64)
                .to_numpy()
                for feature in CATEGORICAL_FEATURES
            ]
        )
        numeric = X[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce").astype(float)
        for feature in NUMERIC_FEATURES:
            numeric[feature] = numeric[feature].fillna(self.numeric_means_[feature])
            numeric[feature] = (
                numeric[feature] - self.numeric_means_[feature]
            ) / self.numeric_stds_[feature]
        return _EncodedFrame(
            categories=np.ascontiguousarray(categories, dtype=np.int64),
            numerics=np.ascontiguousarray(numeric.to_numpy(), dtype=np.float32),
        )

    def _build_network(self) -> Any:
        if self.architecture == "mlp":
            return _EmbeddingMLP(
                self.category_cardinalities_,
                len(NUMERIC_FEATURES),
                tuple(self.hidden_dims),
                self.dropout,
            )
        if self.architecture == "ft_transformer":
            return _FTTransformer(
                self.category_cardinalities_,
                len(NUMERIC_FEATURES),
                self.d_token,
                self.n_heads,
                self.n_layers,
                self.ff_multiplier,
                self.dropout,
            )
        raise ValueError("architecture must be 'mlp' or 'ft_transformer'")

    @staticmethod
    def _tensor_dataset(encoded: _EncodedFrame, y: np.ndarray | None = None) -> Any:
        tensors = [
            torch.from_numpy(encoded.categories),
            torch.from_numpy(encoded.numerics),
        ]
        if y is not None:
            tensors.append(torch.from_numpy(np.asarray(y, dtype=np.float32)))
        return TensorDataset(*tensors)

    def fit(self, X: pd.DataFrame, y: Any) -> "TorchTabularClassifier":
        _require_torch()
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        y_array = np.asarray(y, dtype=np.int64)
        if len(X) != len(y_array):
            raise ValueError("X and y have different lengths")
        if len(np.unique(y_array)) < 2:
            raise ValueError("TorchTabularClassifier requires both target classes")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5")

        _seed_everything(self.random_state)
        self.classes_ = np.asarray([0, 1], dtype=np.int64)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self._fit_encoders(X)
        encoded = self._encode(X)

        n_validation = max(1, int(round(len(X) * self.validation_fraction)))
        n_train = len(X) - n_validation
        if n_train < 2:
            raise ValueError("Not enough rows for neural train/validation split")
        train_encoded = _EncodedFrame(
            encoded.categories[:n_train], encoded.numerics[:n_train]
        )
        validation_encoded = _EncodedFrame(
            encoded.categories[n_train:], encoded.numerics[n_train:]
        )
        y_train = y_array[:n_train]
        y_validation = y_array[n_train:]

        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        train_loader = DataLoader(
            self._tensor_dataset(train_encoded, y_train),
            batch_size=min(self.batch_size, n_train),
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        validation_loader = DataLoader(
            self._tensor_dataset(validation_encoded, y_validation),
            batch_size=min(self.batch_size * 2, n_validation),
            shuffle=False,
            num_workers=0,
        )

        self.device_ = self._resolve_device()
        self.network_ = self._build_network().to(self.device_)
        self.parameter_count_ = int(
            sum(parameter.numel() for parameter in self.network_.parameters())
        )
        positives = max(int(y_train.sum()), 1)
        negatives = max(int(len(y_train) - positives), 1)
        pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=self.device_)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self.network_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        best_loss = math.inf
        best_state: dict[str, Any] | None = None
        stale_epochs = 0
        self.training_history_ = []
        for epoch in range(1, self.epochs + 1):
            self.network_.train()
            training_loss = 0.0
            training_rows = 0
            for categories, numerics, targets in train_loader:
                categories = categories.to(self.device_)
                numerics = numerics.to(self.device_)
                targets = targets.to(self.device_)
                optimizer.zero_grad(set_to_none=True)
                logits = self.network_(categories, numerics)
                loss = criterion(logits, targets)
                loss.backward()
                if self.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.network_.parameters(), self.gradient_clip_norm)
                optimizer.step()
                training_loss += float(loss.detach().cpu()) * len(targets)
                training_rows += len(targets)

            self.network_.eval()
            validation_loss = 0.0
            validation_rows = 0
            with torch.no_grad():
                for categories, numerics, targets in validation_loader:
                    categories = categories.to(self.device_)
                    numerics = numerics.to(self.device_)
                    targets = targets.to(self.device_)
                    logits = self.network_(categories, numerics)
                    loss = criterion(logits, targets)
                    validation_loss += float(loss.detach().cpu()) * len(targets)
                    validation_rows += len(targets)
            train_mean = training_loss / max(training_rows, 1)
            validation_mean = validation_loss / max(validation_rows, 1)
            self.training_history_.append(
                {"epoch": epoch, "train_loss": train_mean, "validation_loss": validation_mean}
            )
            if self.verbose:
                print(
                    f"[{self.architecture}] epoch={epoch} "
                    f"train_loss={train_mean:.5f} validation_loss={validation_mean:.5f}"
                )
            if validation_mean < best_loss - self.min_delta:
                best_loss = validation_mean
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.network_.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        if best_state is None:  # pragma: no cover - defensive
            raise RuntimeError("Neural training did not produce a valid checkpoint")
        self.network_.load_state_dict(best_state)
        self.network_.to("cpu")
        self.device_ = "cpu"
        self.best_validation_loss_ = float(best_loss)
        self.n_epochs_trained_ = len(self.training_history_)
        return self

    def _raw_logits(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["network_", "category_maps_", "numeric_means_"])
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        encoded = self._encode(X)
        loader = DataLoader(
            self._tensor_dataset(encoded),
            batch_size=min(self.batch_size * 2, max(len(X), 1)),
            shuffle=False,
            num_workers=0,
        )
        outputs: list[np.ndarray] = []
        self.network_.eval()
        with torch.no_grad():
            for categories, numerics in loader:
                logits = self.network_(categories, numerics)
                outputs.append(logits.detach().cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty(0, dtype=float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        logits = self._raw_logits(X)
        positive = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        return self._raw_logits(X)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        network = state.get("network_")
        if network is not None:
            network.to("cpu")
            state["device_"] = "cpu"
        return state
