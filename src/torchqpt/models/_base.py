import os
import json
from abc import ABCMeta, abstractmethod
from typing import Any

import torch
from torch import nn

COMPLEX_DTYPE = torch.complex64

class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs

    @abstractmethod
    def forward(self, *args: Any, **kwds: Any):
        pass

    def save(self, path, tag=""):
        if not os.path.exists(path):
            os.makedirs(path)
        args = json.dumps(self.kwargs)
        with open(os.path.join(path, "model_args.json"), "w") as f:
            f.write(args)
        f.close()
        torch.save(self.state_dict(), os.path.join(path, f"model_{tag}.pt"))

    @classmethod
    def load(cls, path, tag="final"):
        with open(os.path.join(path, "model_args.json"), "r") as f:
            kwargs = json.load(f)
        f.close()
        model = cls(**kwargs)
        model.load_state_dict(torch.load(os.path.join(path, f"model_{tag}.pt"),map_location='cpu'))
        return model

    def flops(self, trainable=True, formatted=True) -> int:
        """
        ## Calculate the number of FLOPs for the model.

        ### Args:
            - trainable: whether to count the trainable parameters, Default = True
                If False, count all parameters.
            - formatted: whether to format the number, Default = True
                If False, return the raw number.

        ### Returns:
            - flops: number of FLOPs
            - str: formatted string with appropriate suffix
        """
        if not trainable:
            flops = sum(p.numel() for p in self.parameters())
        else:
            flops = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if formatted:
            flops = self.__format_number(flops)
        return flops

    def __format_number(self, num, precision=3):
        """
        ## Auto-format model size in K, M, G, or T units.

        ### Args:
            - num: number to format
            - precision: number of decimal places to keep, Default = 3
        """
        units = [("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)]
        for suffix, threshold in units:
            if num >= threshold:
                return f"{num / threshold:.{precision}f}{suffix}"
        return f"{num:.{precision}f}"

    def __str__(self):
        _str = super().__str__()
        _str += (
            f"\nModel FLOPS: {self.flops()}, Params: {self.flops(trainable=False)}\n"
        )
        _str += f"Model Args:\n{json.dumps(self.kwargs, indent=4)}\n"
        return _str

    def model_train(self, *args: Any, **kwds: Any):
        pass

    def model_eval(self, *args: Any, **kwds: Any):
        pass
