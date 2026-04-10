# -*- coding: utf-8 -*-
from typing import Optional, Tuple, Union, List

import logging

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


SUPPORTED_SURGICAL_FEAT_TYPES = Union[List, pd.DataFrame, np.ndarray]
SUPPORTED_SURGICAL_TARGET_TYPES = Union[List, pd.Series, pd.DataFrame, np.ndarray, spmatrix]


def convert_if_sparse(
    y: SUPPORTED_SURGICAL_TARGET_TYPES,
) -> Union[np.ndarray, List, pd.DataFrame, pd.Series]:
    if isinstance(y, spmatrix):
        y_ = y.toarray()
        if y_.ndim == 2 and y_.shape[1] == 1:
            y_ = y_.ravel()
    else:
        y_ = y
    return y_


class SurgicalFeatureValidator:
    def __init__(
        self,
        allow_nan: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.allow_nan = allow_nan
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        self.data_type = None
        self.feature_dim = None
        self.sequence_length = None
        self._is_fitted = False

    def fit(
        self,
        X_train: SUPPORTED_SURGICAL_FEAT_TYPES,
        X_test: Optional[SUPPORTED_SURGICAL_FEAT_TYPES] = None,
    ) -> "SurgicalFeatureValidator":
        X_train = self._to_numpy(X_train)
        self._check_data(X_train, is_train=True)

        if X_test is not None:
            X_test = self._to_numpy(X_test)
            self._check_data(X_test, is_train=False)

        self._is_fitted = True
        return self

    def transform(
        self,
        X: SUPPORTED_SURGICAL_FEAT_TYPES,
    ) -> np.ndarray:
        if not self._is_fitted:
            raise NotFittedError("Cannot call transform on a validator that is not fitted")

        X = self._to_numpy(X)
        self._check_data(X, is_train=False)
        return X.astype(np.float32)

    def _to_numpy(self, X: SUPPORTED_SURGICAL_FEAT_TYPES) -> np.ndarray:
        if isinstance(X, list):
            return np.asarray(X, dtype=np.float32)
        elif isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=np.float32)
        elif isinstance(X, np.ndarray):
            return X
        else:
            raise ValueError(f"Unsupported feature type: {type(X)}")

    def _check_data(self, X: np.ndarray, is_train: bool = True) -> None:
        if not isinstance(X, np.ndarray):
            raise ValueError(f"Feature input must be numpy.ndarray, got {type(X)}")

        if X.ndim not in (2, 3):
            raise ValueError(
                f"Surgical feature input must be 2D or 3D, got shape={X.shape}"
            )

        if not np.issubdtype(X.dtype, np.number):
            raise ValueError(f"Feature input must be numeric, got dtype={X.dtype}")

        if not self.allow_nan and np.isnan(X).any():
            raise ValueError("Feature input contains NaN, but allow_nan=False")

        if is_train:
            self.data_type = type(X)
            if X.ndim == 3:
                self.sequence_length = X.shape[1]
                self.feature_dim = X.shape[2]
            else:
                self.sequence_length = 1
                self.feature_dim = X.shape[1]
        else:
            if X.ndim == 3 and X.shape[2] != self.feature_dim:
                raise ValueError(
                    f"Feature dimension mismatch: train={self.feature_dim}, current={X.shape[2]}"
                )
            if X.ndim == 2 and X.shape[1] != self.feature_dim:
                raise ValueError(
                    f"Feature dimension mismatch: train={self.feature_dim}, current={X.shape[1]}"
                )


class SurgicalTargetValidator:
    def __init__(
        self,
        task_type: str = "sequence_classification",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.task_type = task_type
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        self.classes_ = None
        self.class_to_index = {}
        self.index_to_class = {}
        self._is_fitted = False

    def fit(
        self,
        y_train: SUPPORTED_SURGICAL_TARGET_TYPES,
        y_test: Optional[SUPPORTED_SURGICAL_TARGET_TYPES] = None,
    ) -> "SurgicalTargetValidator":
        y_train = self._to_numpy(y_train)
        self._check_data(y_train)

        if y_test is not None:
            y_test = self._to_numpy(y_test)
            self._check_data(y_test)

        all_y = y_train.reshape(-1)
        if y_test is not None:
            all_y = np.concatenate([all_y, y_test.reshape(-1)], axis=0)

        self.classes_ = np.unique(all_y)
        self.class_to_index = {c: i for i, c in enumerate(self.classes_)}
        self.index_to_class = {i: c for i, c in enumerate(self.classes_)}

        self._is_fitted = True
        return self

    def transform(self, y: SUPPORTED_SURGICAL_TARGET_TYPES) -> np.ndarray:
        if not self._is_fitted:
            raise NotFittedError("Cannot call transform on a validator that is not fitted")

        y = self._to_numpy(y)
        self._check_data(y)

        if y.ndim == 1:
            return np.array([self.class_to_index[v] for v in y], dtype=np.int64)

        out = np.empty_like(y, dtype=np.int64)
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                out[i, j] = self.class_to_index[y[i, j]]
        return out

    def inverse_transform(self, y: SUPPORTED_SURGICAL_TARGET_TYPES) -> np.ndarray:
        if not self._is_fitted:
            raise NotFittedError("Cannot call inverse_transform on a validator that is not fitted")

        y = self._to_numpy(y)

        if y.ndim == 1:
            return np.array([self.index_to_class[int(v)] for v in y], dtype=object)

        out = np.empty_like(y, dtype=object)
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                out[i, j] = self.index_to_class[int(y[i, j])]
        return out

    def _to_numpy(self, y: SUPPORTED_SURGICAL_TARGET_TYPES) -> np.ndarray:
        y = convert_if_sparse(y)
        if isinstance(y, list):
            return np.asarray(y)
        elif isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            return y.to_numpy()
        elif isinstance(y, np.ndarray):
            return y
        else:
            raise ValueError(f"Unsupported target type: {type(y)}")

    def _check_data(self, y: np.ndarray) -> None:
        if self.task_type == "sequence_classification":
            if y.ndim != 1:
                raise ValueError(
                    f"For sequence classification, y must be 1D, got {y.shape}"
                )
        elif self.task_type == "frame_classification":
            if y.ndim != 2:
                raise ValueError(
                    f"For frame classification, y must be 2D, got {y.shape}"
                )
        else:
            raise ValueError(
                "task_type must be 'sequence_classification' or 'frame_classification'"
            )

        if np.issubdtype(y.dtype, np.number) and np.isnan(y).any():
            raise ValueError("Target values cannot contain NaN")


class SurgicalInputValidator(BaseEstimator):
    """
    适用于手术视频特征与动作识别的输入验证器

    支持:
    1. 片段级分类:
       X -> (N, T, D)
       y -> (N,)

    2. 逐帧分类:
       X -> (N, T, D)
       y -> (N, T)
    """

    def __init__(
        self,
        task_type: str = "sequence_classification",
        allow_nan_features: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.task_type = task_type
        self.logger = logger if logger is not None else logging.getLogger("SurgicalValidation")

        self.feature_validator = SurgicalFeatureValidator(
            allow_nan=allow_nan_features,
            logger=self.logger,
        )
        self.target_validator = SurgicalTargetValidator(
            task_type=task_type,
            logger=self.logger,
        )
        self._is_fitted = False

    def fit(
        self,
        X_train: SUPPORTED_SURGICAL_FEAT_TYPES,
        y_train: SUPPORTED_SURGICAL_TARGET_TYPES,
        X_test: Optional[SUPPORTED_SURGICAL_FEAT_TYPES] = None,
        y_test: Optional[SUPPORTED_SURGICAL_TARGET_TYPES] = None,
    ) -> BaseEstimator:
        X_train = self.feature_validator._to_numpy(X_train)
        y_train = self.target_validator._to_numpy(y_train)

        if self.task_type == "sequence_classification":
            if X_train.shape[0] != y_train.shape[0]:
                raise ValueError(
                    f"Inconsistent train sample count: X={X_train.shape[0]}, y={y_train.shape[0]}"
                )
        else:
            if X_train.shape[0] != y_train.shape[0] or X_train.shape[1] != y_train.shape[1]:
                raise ValueError(
                    f"Inconsistent train shape for frame classification: X={X_train.shape}, y={y_train.shape}"
                )

        if X_test is not None and y_test is not None:
            X_test = self.feature_validator._to_numpy(X_test)
            y_test = self.target_validator._to_numpy(y_test)

            if self.task_type == "sequence_classification":
                if X_test.shape[0] != y_test.shape[0]:
                    raise ValueError(
                        f"Inconsistent test sample count: X={X_test.shape[0]}, y={y_test.shape[0]}"
                    )
            else:
                if X_test.shape[0] != y_test.shape[0] or X_test.shape[1] != y_test.shape[1]:
                    raise ValueError(
                        f"Inconsistent test shape for frame classification: X={X_test.shape}, y={y_test.shape}"
                    )

        self.feature_validator.fit(X_train, X_test)
        self.target_validator.fit(y_train, y_test)
        self._is_fitted = True
        return self

    def transform(
        self,
        X: SUPPORTED_SURGICAL_FEAT_TYPES,
        y: Optional[SUPPORTED_SURGICAL_TARGET_TYPES] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self._is_fitted:
            raise NotFittedError("Cannot call transform on a validator that is not fitted")

        X_transformed = self.feature_validator.transform(X)

        if y is not None:
            y_transformed = self.target_validator.transform(y)
            return X_transformed, y_transformed

        return X_transformed, None