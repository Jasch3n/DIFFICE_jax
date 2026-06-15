from typing import NamedTuple

from jax.typing import ArrayLike


class FieldState(NamedTuple):
    u: ArrayLike
    v: ArrayLike
    h: ArrayLike
    s: ArrayLike | None = None
    mu: ArrayLike | None = None
    C: ArrayLike | None = None
    k: ArrayLike | None = None


class FieldDerivatives(NamedTuple):
    u_x: ArrayLike
    u_y: ArrayLike
    v_x: ArrayLike
    v_y: ArrayLike
    h_x: ArrayLike
    h_y: ArrayLike
    s_x: ArrayLike | None = None
    s_y: ArrayLike | None = None
    mu_x: ArrayLike | None = None
    mu_y: ArrayLike | None = None
    C_x: ArrayLike | None = None
    C_y: ArrayLike | None = None
