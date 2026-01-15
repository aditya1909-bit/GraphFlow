from __future__ import annotations

"""
A tiny reverse‑mode autograd Tensor for GraphFlow.

Supports:
- elementwise: +, -, *, /, ** (pow), neg
- matrix: matmul (@), transpose (T)
- reductions: sum, mean (with axis + keepdims)
- activations: relu, exp, log
- backprop: .backward()

Notes
-----
This is intentionally minimal and NumPy‑only. Gradients are accumulated in
`Tensor.grad` and cleared by optimizers after an update.
"""

from typing import Callable, Iterable, Optional, Tuple, List, Union
import numpy as np

Array = np.ndarray


def _as_tensor(x: Union['Tensor', Array, float, int]) -> 'Tensor':
    return x if isinstance(x, Tensor) else Tensor(x)


def _expand_grad(grad: Array, in_shape: Tuple[int, ...], axis, keepdims: bool) -> Array:
    """Expand a reduced gradient back to input shape for broadcasting.

    Handles axis being None / int / tuple[int,...].
    """
    if axis is None:
        return np.ones(in_shape, dtype=np.float64) * grad
    # normalize axis to tuple of positive indices
    if isinstance(axis, int):
        axes = (axis,)
    else:
        axes = tuple(axis)
    axes = tuple(a if a >= 0 else a + len(in_shape) for a in axes)
    g = grad
    if not keepdims:
        # Insert singleton dims at the reduced axes
        for ax in sorted(axes):
            g = np.expand_dims(g, ax)
    # Broadcast to input shape
    return np.broadcast_to(g, in_shape)


def _sum_to_shape(grad: Array, shape: Tuple[int, ...]) -> Array:
    """Sum gradients to match a target shape (for broadcasting)."""
    g = np.asarray(grad)
    if shape == ():
        return np.array(g).sum()
    if g.shape == shape:
        return g
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for axis, (gdim, sdim) in enumerate(zip(g.shape, shape)):
        if sdim == 1 and gdim != 1:
            g = g.sum(axis=axis, keepdims=True)
    if g.shape != shape:
        g = g.reshape(shape)
    return g


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "name")

    def __init__(self, data, requires_grad: bool = False, name: Optional[str] = None):
        self.data: Array = np.asarray(data, dtype=np.float64)
        self.grad: Optional[Array] = None
        self.requires_grad: bool = bool(requires_grad)
        self._backward: Callable[[], None] = lambda: None
        self._prev: Tuple['Tensor', ...] = tuple()
        self.name = name

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        rg = "requires_grad=True" if self.requires_grad else "requires_grad=False"
        return f"Tensor(shape={self.data.shape}, {rg}, name={self.name})"

    def _binary_out(self, other: 'Tensor', data: Array) -> 'Tensor':
        out = Tensor(data, self.requires_grad or other.requires_grad)
        out._prev = (self, other)
        return out

    # ------------------------------------------------------------------
    # Basic arithmetic
    # ------------------------------------------------------------------
    def __add__(self, other):
        other = _as_tensor(other)
        out = self._binary_out(other, self.data + other.data)

        def _backward():
            if self.requires_grad:
                self._add_grad(out.grad)
            if other.requires_grad:
                other._add_grad(out.grad)
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = _as_tensor(other)
        out = self._binary_out(other, self.data - other.data)

        def _backward():
            if self.requires_grad:
                self._add_grad(out.grad)
            if other.requires_grad:
                other._add_grad(-out.grad)
        out._backward = _backward
        return out

    def __rsub__(self, other):
        other = _as_tensor(other)
        return other.__sub__(self)

    def __neg__(self):
        out = Tensor(-self.data, self.requires_grad)

        def _backward():
            if self.requires_grad:
                self._add_grad(-out.grad)
        out._prev = (self,)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = _as_tensor(other)
        out = self._binary_out(other, self.data * other.data)

        def _backward():
            if self.requires_grad:
                self._add_grad(out.grad * other.data)
            if other.requires_grad:
                other._add_grad(out.grad * self.data)
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = _as_tensor(other)
        return self * other.pow( -1.0)

    def __rtruediv__(self, other):
        other = _as_tensor(other)
        return other * self.pow( -1.0)

    def pow(self, p: float) -> 'Tensor':
        out = Tensor(self.data ** p, self.requires_grad)

        def _backward():
            if self.requires_grad:
                self._add_grad(out.grad * (p * (self.data ** (p - 1))))
        out._prev = (self,)
        out._backward = _backward
        return out

    def __pow__(self, p: float):
        return self.pow(p)

    # ------------------------------------------------------------------
    # Linear algebra
    # ------------------------------------------------------------------
    def matmul(self, other: 'Tensor') -> 'Tensor':
        other = _as_tensor(other)
        out = self._binary_out(other, self.data @ other.data)

        def _backward():
            if self.requires_grad:
                self._add_grad(out.grad @ other.data.T)
            if other.requires_grad:
                other._add_grad(self.data.T @ out.grad)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def T(self) -> 'Tensor':
        out = Tensor(self.data.T, self.requires_grad)

        def _backward():
            if self.requires_grad:
                self._add_grad(out.grad.T)
        out._prev = (self,)
        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Reductions & elementwise
    # ------------------------------------------------------------------
    def sum(self, axis=None, keepdims: bool = False) -> 'Tensor':
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad)

        def _backward():
            if self.requires_grad:
                g = _expand_grad(out.grad, self.data.shape, axis, keepdims)
                self._add_grad(g)
        out._prev = (self,)
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False) -> 'Tensor':
        if axis is None:
            denom = self.data.size
        else:
            if isinstance(axis, int):
                denom = self.data.shape[axis]
            else:
                denom = np.prod([self.data.shape[a] for a in axis])
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / float(denom))

    def relu(self) -> 'Tensor':
        mask = self.data > 0
        out = Tensor(self.data * mask, self.requires_grad)

        def _backward():
            if self.requires_grad:
                self._add_grad(out.grad * mask)
        out._prev = (self,)
        out._backward = _backward
        return out

    def exp(self) -> 'Tensor':
        e = np.exp(self.data)
        out = Tensor(e, self.requires_grad)

        def _backward():
            if self.requires_grad:
                self._add_grad(out.grad * e)
        out._prev = (self,)
        out._backward = _backward
        return out

    def log(self) -> 'Tensor':
        out = Tensor(np.log(self.data + 1e-12), self.requires_grad)

        def _backward():
            if self.requires_grad:
                self._add_grad(out.grad * (1.0 / (self.data + 1e-12)))
        out._prev = (self,)
        out._backward = _backward
        return out

    def sigmoid(self) -> 'Tensor':
        out = Tensor(1.0 / (1.0 + np.exp(-self.data)), self.requires_grad)

        def _backward():
            if self.requires_grad:
                self._add_grad(out.grad * (out.data * (1.0 - out.data)))
        out._prev = (self,)
        out._backward = _backward
        return out

    def reshape(self, shape) -> 'Tensor':
        out = Tensor(self.data.reshape(shape), self.requires_grad)

        def _backward():
            if self.requires_grad:
                self._add_grad(out.grad.reshape(self.data.shape))
        out._prev = (self,)
        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Autodiff
    # ------------------------------------------------------------------
    def _add_grad(self, g: Optional[Array]):
        if g is None:
            return
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad = self.grad + _sum_to_shape(g, self.data.shape)

    def backward(self, grad: Optional[Array] = None):
        """Backpropagate from this tensor.

        If this tensor is a scalar (size==1), the external gradient defaults to 1.
        For non-scalars, a gradient must be provided.
        """
        if grad is None:
            if self.data.size != 1:
                raise ValueError("grad must be specified for non-scalar outputs")
            grad = np.ones_like(self.data)

        topo: List[Tensor] = []
        visited = set()

        def build(v: 'Tensor'):
            if id(v) not in visited:
                visited.add(id(v))
                for p in v._prev:
                    build(p)
                topo.append(v)

        build(self)
        self.grad = grad
        for v in reversed(topo):
            v._backward()
