# this module is used as a base class for cross validation datasets
import sys
from abc import ABC, abstractmethod
from monai.data import CacheDataset


@abstractmethod
class CVDataset(ABC, CacheDataset):
    """
    Base class to generate cross validation datasets.

    """
    def __init__(
        self,
        data,
        transform,
        cache_num=sys.maxsize,
        cache_rate=1.0,
        num_workers=2,
    ) -> None:
        data = self._split_datalist(datalist=data)
        CacheDataset.__init__(
            self, data, transform, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers
        )

    def _split_datalist(self, datalist):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")