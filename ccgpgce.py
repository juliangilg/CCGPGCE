import tensorflow as tf
import gpflow as gpf

from gpflow.ci_utils import reduce_in_tests
from gpflow.utilities import print_summary
from gpflow.quadrature import hermgauss
from gpflow.quadrature.deprecated import mvnquad, ndiagquad, ndiag_mc
from check_shapes import check_shapes, inherit_check_shapes



class multiClassMA_GCE(gpf.likelihoods.Likelihood):
    def __init__(
        self, num_classes: int, num_ann: int, q: int) -> None:
        super().__init__(input_dim=None, latent_dim=num_classes+num_ann, observation_dim=None)
        self.K = num_classes
        self.R = num_ann
        self.q = q

    @inherit_check_shapes
    def _log_prob(self, X, F, Y):

        iAnn = tf.where(Y == -1e20, tf.zeros_like(Y), tf.ones_like(Y))
        Yh   = tf.one_hot(tf.cast(Y-1, tf.int32), depth= self.K, axis=1)
        Yh   = tf.cast(Yh, tf.float64)
        zeta = tf.repeat(tf.expand_dims((F[:,:self.K]), axis = -1), self.R, axis=-1)
        lamb = tf.nn.sigmoid(F[:,self.K:])

        CE = -tf.math.reduce_sum(Yh*self.log_softmax(zeta), axis=1)

        return -tf.math.reduce_sum((lamb*CE + (1-lamb)*(1 - (self.K)**(self.q))), axis=1)/self.q

    def _variational_expectations(self, X, Fmu, Fvar, Y):

        iAnn = tf.where(Y == -1e20, tf.zeros_like(Y), tf.ones_like(Y))
        m_f, m_g = Fmu[:, :self.K], Fmu[:, self.K:]
        v_f, v_g = Fvar[:, :self.K], Fvar[:, self.K:]

        iAnn = tf.where(Y == -1e20, tf.zeros_like(Y), tf.ones_like(Y))
        Yh   = tf.one_hot(tf.cast(Y-1, tf.int32), depth= self.K, axis=1)
        Yh   = tf.cast(Yh, tf.float64)


        # E_{q(f_{1,n})...q(f_{K,n})}[log zeta]
        Elog  = ndiag_mc(self.log_softmax, 100, m_f, v_f, False)# E[log(softmax(F))]
        Elog_ = tf.repeat(tf.expand_dims(Elog, axis = -1), self.R, axis=-1)
        C_E   = -tf.math.reduce_sum((Yh*Elog_), axis = 1)


        # E_{q(g_m^m)}[z_n^m]
        Eq_g = ndiagquad(tf.nn.sigmoid, 20, m_g, v_g, False)


        #Variational Expectation ##########################################
        return tf.math.reduce_sum((Eq_g*C_E + (1-Eq_g)*(1 - self.K**(self.q))), axis=1)/self.q

    def _predict_mean_and_var(self, X, Fmu, Fvar):
        m_f, m_g = Fmu[:, :self.K], Fmu[:, self.K:]
        v_f, v_g = Fvar[:, :self.K], Fvar[:, self.K:]

        Ez = ndiag_mc(tf.nn.softmax, 800, m_f, v_f, False)# E[(softmax(F))]
        El = ndiagquad(tf.nn.sigmoid, 20, m_g, v_g, False)

        Ez_2 = ndiag_mc(self.softmax_2, 800, m_f, v_f, False)# E[(softmax(F))]
        El_2 = ndiagquad(self.sigmoid_2, 20, m_g, v_g, False)

        return tf.concat([Ez, El], axis = 1), tf.concat([Ez_2-Ez**2, El_2-El**2], axis = 1)

    def _predict_log_density(self, F):
        raise NotImplementedError

    def log_softmax(self, F):
        return (1 - (tf.nn.softmax(F))**(self.q))

    def softmax_2(self, F):
        return tf.math.square(tf.nn.softmax(F))

    def sigmoid_2(self, F):
        return tf.math.square(tf.nn.sigmoid(F))
