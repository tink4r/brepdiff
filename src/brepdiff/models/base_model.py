from torch import nn as nn
from abc import ABC, abstractmethod

from brepdiff.utils.acc_logger import AccLogger
from brepdiff.config import Config

# ==========
# Model ABCs
# ==========


class BaseModule(nn.Module, ABC):
    def __init__(self, config: Config, acc_logger: AccLogger):
        super().__init__()
        self.config = config
        self.acc_logger = acc_logger


class BaseModel(BaseModule):
    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError()
