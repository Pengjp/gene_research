import tensorflow as tf
from tf_poly import TfPoly
import numpy as np


def runif():
    return tf.random.uniform([1], dtype=tf.float64)[0]


def rexp():
    return -tf.math.log(runif())


def gen_gaps(
    k: int,
    eta_x: tf.Tensor,
    eta_c: tf.Tensor,
    theta=tf.constant(1e-4, dtype=tf.float64),
    rho=tf.constant(1e-5, dtype=tf.float64),
) -> tf.Tensor:
    "Return k gaps sampled from genetic distribution with rate function eta."
    eta = TfPoly(x=eta_x, c=eta_c)
    R = eta.antiderivative()
    Rinv = R.inverse()
    x = Rinv(rexp())  # initialize x by sampling from prior
    pos = tf.constant(0., dtype=tf.float64)
    j = 0
    ta = tf.TensorArray(tf.float64, size=k + 2)
    while tf.less(j, k + 2):
        # x' satisfies R(x') - R(u*x) = Z => x' = Rinv(Z + R(u*x))
        u = runif()
        z = rexp()
        x = Rinv(z + R(u * x))  # segment height
        pos += rexp() / (x * (theta + rho))  # length to next event
        while runif() < (theta / (theta + rho)) and tf.less(j, k + 2):
            ta = ta.write(j, pos)
            j += 1
            pos += rexp() / (x * (theta + rho))  # length to next event
    ret = ta.stack()[1:]  # first obs suffers from inspection paradox?
    return ret[1:] - ret[:-1]
