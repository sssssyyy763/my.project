# -*- coding: utf-8 -*-
from typing import List, Optional, Union, Dict, Any

import logging
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


SUPPORTED_TARGET_TYPES = Union[List, pd.Series, pd.DataFrame, np.ndarray, spmatrix]


class SurgicalTargetValidator(BaseEstimator):
    """
    适用于手术视频动作识别的标签验证器

    支持两类任务：
    1. sequence_classification
       - 每个视频片段一个标签
       - y shape: (N,)

    2. frame_classification
       - 每个视频片段中每一帧一个标签
       - y shape: (N, T)

    功能：
    - 检查标签格式
    - 建立类别到整数的映射
    - transform/inverse_transform
    """

    def __init__(
        self,
        task_type: str = "sequence_classification",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.task_type = task_type
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        self.data_type = None
        self.classes_: Optional[np.ndarray] = None
        self.class_to_index: Dict[Any, int] = {}
        self.index_to_class: Dict[int, Any] = {}

        self.out_dimensionality = None
        self._is_fitted = False

    def fit(
        self,
        y_train: SUPPORTED_TARGET_TYPES,
        y_test: Optional[SUPPORTED_TARGET_TYPES] = None,
    ) -> "SurgicalTargetValidator":
        y_train = self._to_numpy(y_train)
        self._check_data(y_train, is_train=True)

        if y_test is not None:
            y_test = self._to_numpy(y_test)
            self._check_data(y_test, is_train=False)

            if y_train.ndim != y_test.ndim:
                raise ValueError(
                    f"Train/Test target dimensionality mismatch: "
                    f"{y_train.shape} vs {y_test.shape}"
                )

            if y_train.ndim == 2 and y_train.shape[1] != y_test.shape[1]:
                raise ValueError(
                    f"Train/Test sequence length mismatch: "
                    f"{y_train.shape} vs {y_test.shape}"
                )

        if y_train.ndim == 1:
            self.out_dimensionality = 1
        else:
            self.out_dimensionality = y_train.shape[1]

        if y_test is not None:
            all_y = np.concatenate([y_train.reshape(-1), y_test.reshape(-1)], axis=0)
        else:
            all_y = y_train.reshape(-1)

        classes = np.unique(all_y)
        self.classes_ = classes
        self.class_to_index = {c: i for i, c in enumerate(classes)}
        self.index_to_class = {i: c for i, c in enumerate(classes)}

        self._is_fitted = True
        return self

    def transform(self, y: SUPPORTED_TARGET_TYPES) -> np.ndarray:
        if not self._is_fitted:
            raise NotFittedError("SurgicalTargetValidator must have fit() called first")

        y = self._to_numpy(y)
        self._check_data(y, is_train=False)

        if y.ndim == 1:
            return np.array([self.class_to_index[v] for v in y], dtype=np.int64)
        else:
            out = np.empty_like(y, dtype=np.int64)
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    out[i, j] = self.class_to_index[y[i, j]]
            return out

    def inverse_transform(self, y: SUPPORTED_TARGET_TYPES) -> np.ndarray:
        if not self._is_fitted:
            raise NotFittedError("SurgicalTargetValidator must have fit() called first")

        y = self._to_numpy(y)

        if y.ndim == 1:
            return np.array([self.index_to_class[int(v)] for v in y], dtype=object)
        else:
            out = np.empty_like(y, dtype=object)
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    out[i, j] = self.index_to_class[int(y[i, j])]
            return out

    def validate_with_X(
        self,
        X: np.ndarray,
        y: SUPPORTED_TARGET_TYPES,
    ) -> np.ndarray:
        """
        检查标签和输入特征是否匹配

        sequence_classification:
            X: (N, T, D), y: (N,)
        frame_classification:
            X: (N, T, D), y: (N, T)
        """
        y = self._to_numpy(y)

        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy.ndarray")

        if X.ndim != 3:
            raise ValueError(
                f"For surgical video action recognition, X must be 3D (N, T, D), got {X.shape}"
            )

        if self.task_type == "sequence_classification":
            if y.ndim != 1:
                raise ValueError(
                    f"For sequence classification, y must be 1D, got shape={y.shape}"
                )
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"Sample count mismatch: X has {X.shape[0]} samples but y has {y.shape[0]}"
                )

        elif self.task_type == "frame_classification":
            if y.ndim != 2:
                raise ValueError(
                    f"For frame classification, y must be 2D, got shape={y.shape}"
                )
            if X.shape[0] != y.shape[0] or X.shape[1] != y.shape[1]:
                raise ValueError(
                    f"X and y shape mismatch for frame classification: X={X.shape}, y={y.shape}"
                )
        else:
            raise ValueError(
                "task_type must be 'sequence_classification' or 'frame_classification'"
            )

        return y

    def is_single_column_target(self) -> bool:
        return self.out_dimensionality == 1

    def _to_numpy(self, y: SUPPORTED_TARGET_TYPES) -> np.ndarray:
        if isinstance(y, list):
            return np.asarray(y)
        elif isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            return y.to_numpy()
        elif isinstance(y, spmatrix):
            return y.toarray()
        elif isinstance(y, np.ndarray):
            return y
        else:
            raise ValueError(f"Unsupported target type: {type(y)}")

    def _check_data(self, y: np.ndarray, is_train: bool = True) -> None:
        if self.data_type is None:
            self.data_type = type(y)
        elif self.data_type != type(y):
            self.logger.warning(
                f"Target type changed from {self.data_type} to {type(y)}"
            )

        if np.isnan(y).any() if np.issubdtype(y.dtype, np.number) else False:
            raise ValueError("Target values cannot contain NaN")

        if self.task_type == "sequence_classification":
            if y.ndim != 1:
                raise ValueError(
                    f"For sequence classification, y must be 1D, got shape={y.shape}"
                )

        elif self.task_type == "frame_classification":
            if y.ndim != 2:
                raise ValueError(
                    f"For frame classification, y must be 2D, got shape={y.shape}"
                )

        else:
            raise ValueError(
                "task_type must be 'sequence_classification' or 'frame_classification'"
            )