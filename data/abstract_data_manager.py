# -*- coding: utf-8 -*-
import abc
from typing import Any, Dict, Union

import numpy as np
import scipy.sparse


class AbstractDataManager(metaclass=abc.ABCMeta):
    def __init__(self, name: str):
        self._data = dict()   # type: Dict
        self._info = dict()   # type: Dict
        self._name = name
        self._feat_type = dict()   # type: Dict[Union[str, int], str]
        self._encoder = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def data(self) -> Dict[str, np.ndarray]:
        return self._data

    @property
    def info(self) -> Dict[str, Any]:
        return self._info

    @property
    def feat_type(self) -> Dict[Union[str, int], str]:
        return self._feat_type

    @feat_type.setter
    def feat_type(self, value: Dict[Union[str, int], str]) -> None:
        self._feat_type = value

    @property
    def encoder(self) -> Any:
        return self._encoder

    @encoder.setter
    def encoder(self, value: Any) -> None:
        self._encoder = value

    def __repr__(self) -> str:
        return "DataManager : " + self.name

    def __str__(self) -> str:
        val = "DataManager : " + self.name + "\ninfo:\n"
        for item in self.info:
            val += "\t" + item + " = " + str(self.info[item]) + "\n"
        val += "data:\n"

        for subset in self.data:
            obj = self.data[subset]
            shape = getattr(obj, "shape", "N/A")
            dtype = getattr(obj, "dtype", "N/A")
            val += "\t%s = %s %s %s\n" % (
                subset,
                type(obj),
                str(shape),
                str(dtype),
            )
            if isinstance(obj, scipy.sparse.spmatrix):
                val += "\tdensity: %f\n" % (
                    float(len(obj.data)) / obj.shape[0] / obj.shape[1]
                )
        val += "feat_type:\t" + str(self.feat_type) + "\n"
        return val