"""Minimal optimizers for GraphFlow."""
from __future__ import annotations

from typing import Iterable, Dict
import numpy as np

from .tensor import Tensor


class SGD:
    """In-place SGD with gradient reset."""

    def __init__(self, lr: float = 0.05):
        self.lr = float(lr)

    def step(self, params: Iterable[Tensor]):
        for p in params:
            if p.grad is not None:
                p.data -= self.lr * p.grad
        self.zero_grad(params)

    @staticmethod
    def zero_grad(params: Iterable[Tensor]):
        for p in params:
            if p.grad is not None:
                p.grad = None


class Adam:
    """Adam optimizer with bias correction and gradient reset."""

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.t = 0
        self.state: Dict[int, Dict[str, np.ndarray]] = {}

    def step(self, params: Iterable[Tensor]):
        self.t += 1
        for p in params:
            if p.grad is None:
                continue
            key = id(p)
            if key not in self.state:
                self.state[key] = {
                    "m": np.zeros_like(p.data),
                    "v": np.zeros_like(p.data),
                }
            m = self.state[key]["m"]
            v = self.state[key]["v"]

            m[:] = self.beta1 * m + (1.0 - self.beta1) * p.grad
            v[:] = self.beta2 * v + (1.0 - self.beta2) * (p.grad ** 2)

            m_hat = m / (1.0 - self.beta1 ** self.t)
            v_hat = v / (1.0 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        SGD.zero_grad(params)


__all__ = ["SGD", "Adam"]
