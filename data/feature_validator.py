# -*- coding: utf-8 -*-
from typing import Dict, List, Optional, Tuple, Union, cast

import logging
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_sparse
from scipy.sparse import csr_matrix, spmatrix
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

SUPPORTED_FEAT_TYPES = Union[List, pd.DataFrame, np.ndarray, spmatrix]


class FeatureValidator(BaseEstimator):
    def __init__(
        self,
        feat_type: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
        allow_string_features: bool = True,
    ) -> None:
        self.feat_type: Optional[Dict[Union[str, int], str]] = None

        if feat_type is not None:
            if isinstance(feat_type, dict):
                self.feat_type = feat_type
            elif not isinstance(feat_type, List):
                raise ValueError(
                    "feat_type should be a list or dict, got {}".format(type(feat_type))
                )
            else:
                self.feat_type = {i: feat for i, feat in enumerate(feat_type)}

        self.data_type = None
        self.dtypes = {}
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self._is_fitted = False
        self.allow_string_features = allow_string_features

    def fit(
        self,
        X_train: SUPPORTED_FEAT_TYPES,
        X_test: Optional[SUPPORTED_FEAT_TYPES] = None,
    ) -> "FeatureValidator":
        if isinstance(X_train, List):
            X_train, X_test = self.list_to_dataframe(X_train, X_test)

        self._check_data(X_train)

        if hasattr(X_train, "iloc"):
            if self.feat_type is not None:
                raise ValueError(
                    "When X is DataFrame, feat_type should not be passed explicitly."
                )
            self.feat_type = self.get_feat_type_from_columns(X_train)
        else:
            if self.feat_type is None:
                self.feat_type = {i: "numerical" for i in range(np.shape(X_train)[1])}
            elif len(self.feat_type) != np.shape(X_train)[1]:
                raise ValueError(
                    "feat_type length does not match number of features: %d vs %d"
                    % (len(self.feat_type), np.shape(X_train)[1])
                )

        if X_test is not None:
            self._check_data(X_test)
            if np.shape(X_train)[1] != np.shape(X_test)[1]:
                raise ValueError(
                    "Train/test feature dimension mismatch: %s vs %s"
                    % (np.shape(X_train), np.shape(X_test))
                )

        self._is_fitted = True
        return self

    def transform(
        self,
        X: SUPPORTED_FEAT_TYPES,
    ) -> Union[np.ndarray, spmatrix, pd.DataFrame]:
        if not self._is_fitted:
            raise NotFittedError("FeatureValidator must be fitted first")

        if isinstance(X, List):
            X_transformed, _ = self.list_to_dataframe(X)
        else:
            X_transformed = X

        self._check_data(X_transformed)

        if isinstance(X_transformed, spmatrix):
            if not isinstance(X_transformed, csr_matrix):
                self.logger.warning(
                    "Sparse input is %s, converting to csr_matrix", type(X_transformed)
                )
                X_transformed = X_transformed.tocsr(copy=False)
            X_transformed.sort_indices()

        return X_transformed

    def _check_data(self, X: SUPPORTED_FEAT_TYPES) -> None:
        if hasattr(X, "columns"):
            for column in cast(pd.DataFrame, X).columns:
                if X[column].isna().all():
                    X[column] = X[column].astype("category")

        if not isinstance(X, (np.ndarray, pd.DataFrame)) and not isinstance(X, spmatrix):
            raise ValueError(
                "Only supports numpy.ndarray, pandas.DataFrame, scipy sparse, list"
            )

        if self.data_type is None:
            self.data_type = type(X)
        elif self.data_type != type(X):
            self.logger.warning(
                "Feature type changed from %s to %s", self.data_type, type(X)
            )

        if hasattr(X, "dtype") and isinstance(X, np.ndarray):
            if not np.issubdtype(X.dtype.type, np.number):
                raise ValueError("Numpy array features must be numeric")

        if hasattr(X, "iloc"):
            X = cast(pd.DataFrame, X)
            dtypes = {col: X[col].dtype.name.lower() for col in X.columns}
            if len(self.dtypes) == 0:
                self.dtypes = dtypes
            elif self.dtypes != dtypes:
                self.logger.warning("Feature dtypes changed after fit")

    def get_feat_type_from_columns(
        self,
        X: pd.DataFrame,
    ) -> Dict[Union[str, int], str]:
        feat_type = {}

        for column in X.columns:
            if is_sparse(X[column]):
                raise ValueError(f"Sparse pandas Series not supported: {column}")
            elif X[column].dtype.name in ["category", "bool"]:
                feat_type[column] = "categorical"
            elif X[column].dtype.name == "string":
                feat_type[column] = "string" if self.allow_string_features else "categorical"
            elif not is_numeric_dtype(X[column]):
                if X[column].dtype.name == "object":
                    warnings.warn(
                        f"Column {column} has dtype object, treated as string",
                        UserWarning,
                    )
                    feat_type[column] = "string" if self.allow_string_features else "categorical"
                else:
                    raise ValueError(
                        f"Unsupported dtype in column {column}: {X[column].dtype.name}"
                    )
            else:
                feat_type[column] = "numerical"

        return feat_type

    def list_to_dataframe(
        self,
        X_train: SUPPORTED_FEAT_TYPES,
        X_test: Optional[SUPPORTED_FEAT_TYPES] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        X_train = pd.DataFrame(data=X_train).convert_dtypes()

        if len(self.dtypes) == 0:
            self.dtypes = {
                col: X_train[col].dtype.name.lower() for col in X_train.columns
            }
        else:
            for col in X_train.columns:
                try:
                    X_train[col] = X_train[col].astype(self.dtypes[col])
                except Exception as e:
                    self.logger.warning(
                        "Failed to cast train column %s to %s: %s",
                        col, self.dtypes[col], e
                    )
                    self.dtypes[col] = X_train[col].dtype.name.lower()

        if X_test is not None:
            X_test = pd.DataFrame(data=X_test)
            for col in X_test.columns:
                try:
                    X_test[col] = X_test[col].astype(self.dtypes[col])
                except Exception as e:
                    self.logger.warning(
                        "Failed to cast test column %s to %s: %s",
                        col, self.dtypes[col], e
                    )
                    self.dtypes[col] = X_test[col].dtype.name.lower()

        return X_train, X_test