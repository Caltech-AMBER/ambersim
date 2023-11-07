from ambersim.base import MjxEnv
from dataclasses import dataclass


@dataclass
class A1Config:
    pass


class A1(MjxEnv):
    def __init__(self, **kwargs):
        raise NotImplementedError
