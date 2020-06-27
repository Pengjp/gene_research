from dataclasses import dataclass
import tensorflow as tf
import numpy as np


@dataclass
class TfPoly:
    """Piecewise polynomial, similar to scipy.interpolate.PPoly.

        p(t) = sum_i c[i,j] (t - x[j]) ** i, x[j] <= t < x[j + 1]

    """

    x: tf.Tensor
    c: tf.Tensor

    def __post_init__(self):
        self.x = tf.convert_to_tensor(self.x, dtype=tf.float64)
        self.c = tf.convert_to_tensor(self.c, dtype=tf.float64)
        tf.debugging.assert_rank(self.x, 1)
        tf.debugging.assert_rank(self.c, 2)
        tf.debugging.assert_equal(tf.shape(self.x)[0], 1 + tf.shape(self.c)[1])

    def __call__(self, t):
        "Evaluate p(t)"
        # does not use horners scheme but it's probably ok
        t = tf.convert_to_tensor([t], dtype=tf.float64)
        i = tf.searchsorted(self.x, t, side="right") - 1
        ci = tf.gather(self.c, i, axis=1)  # [D, T]
        ti = t - tf.gather(self.x, i, axis=0)  # [T]
        i = tf.range(ci.shape[0], dtype=tf.float64)[::-1, None]
        return tf.reduce_sum(ci * tf.math.pow(ti[None], i), axis=0)[0]  # [T]
        # return tf.math.polyval(
        #     list(tf.gather(self.c, i, axis=1)), t - tf.gather(self.x, i, axis=0)
        # )

    def antiderivative(self):
        k, n = self.c.shape
        c0 = (1. / tf.range(1, k + 1, dtype=self.c.dtype))[::-1, None] * self.c
        i = 1 + tf.range(c0.shape[0], dtype=c0.dtype)[::-1, None]
        x0 = self.x[1:-1] - self.x[:-2]
        c1 = tf.concat(
            [
                [0.],
                tf.math.cumsum(
                    tf.math.reduce_sum(c0[:, :-1] * tf.math.pow(x0[None], i), axis=0)
                ),
            ],
            axis=0,
        )
        return TfPoly(x=self.x, c=tf.concat([c0, c1[None]], axis=0))

    def inverse(self):
        """Return the inverse of this function. Only valid for continuous,
        strictly monotone, piecewise linear functions; this assumption is not
        checked.
        """
        assert self.c.shape[0] == 2, self.c.shape
        breaks = self.c[1]  # [0, p(t1), p(t2), ...]
        return TfPoly(
            x=tf.concat([breaks, [np.inf]], axis=0),
            c=tf.concat([1. / self.c[:1], [self.x[:-1]]], axis=0),
        )


#### Test code

import scipy
import pytest


@pytest.fixture
def p():
    x = np.r_[0., np.cumsum(np.random.rand(10)), np.inf]
    c = np.random.rand(5, 11)
    return TfPoly(x=x, c=c)


@pytest.fixture
def q(p):
    return scipy.interpolate.PPoly(x=np.array(p.x), c=np.array(p.c))


def test_eval(p, q):
    for t in np.random.rand(10):
        np.testing.assert_allclose(p(t), q(t))


def test_anti(p, q):
    q = scipy.interpolate.PPoly(x=np.array(p.x), c=np.array(p.c))
    R1 = p.antiderivative()
    R2 = q.antiderivative()
    np.testing.assert_allclose(R1.c, R2.c)
    np.testing.assert_allclose(R1.x, R2.x)
    for t in np.random.rand(10):
        np.testing.assert_allclose(R1(t), R2(t))


def test_inverse():
    R = TfPoly(
        x=np.r_[0., np.random.rand(10).cumsum(), np.inf], c=np.random.rand(1, 11)
    ).antiderivative()
    Rinv = R.inverse()
    for t in np.random.rand(10) * 10.:
        np.testing.assert_allclose(R(Rinv(t)), t)
        np.testing.assert_allclose(Rinv(R(t)), t)
