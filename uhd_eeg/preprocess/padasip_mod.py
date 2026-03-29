import numpy as np
import padasip as pa


class AdaptiveFilter(pa.filters.base_filter.AdaptiveFilter):
    """
    Vectorized (and simplified) version of padasip.filters.base_filter.AdaptiveFilter
    """

    def __init__(self, in_ch, out_ch, mu, w="random"):
        """
        Args:
            in_ch (int): number of input channels
            out_ch (int): number of output channels
            mu (float): learning rate
        """
        self.w = self.init_weights(w, in_ch, out_ch)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.mu = mu

    def learning_rule(self, e, x):
        raise NotImplementedError

    def init_weights(self, w, in_ch, out_ch):
        shape = (in_ch, out_ch)
        if isinstance(w, str):
            if w == "random":
                w = np.random.normal(0, 0.5, shape)
            elif w == "zeros":
                w = np.zeros(shape)
            else:
                raise ValueError("Impossible to understand the w")
        elif w.shape == shape:
            try:
                w = np.array(w, dtype="float64")
            except:
                raise ValueError("Impossible to understand the w")
        else:
            raise ValueError("Impossible to understand the w")
        return w

    def predict(self, x):
        """
        Args:
            x (ndarray (in_ch, )): input vector
        Returns:
            y (ndarray (out_ch, )): output vector
        """
        return x @ self.w

    def run(self, d, x):
        """
        Args:
            d (ndarray (n_samp, out_ch)): desired arrays
            x (ndarray (n_samp, in_ch)): input arrays
        """

        y = np.empty_like(d)
        e = np.empty_like(d)
        for k in range(len(d)):
            y[k] = self.predict(x[k])  # (out_ch,)
            e[k] = d[k] - y[k]  # (out_ch,)
            self.w += self.learning_rule(e[k], x[k])  # (in_ch, out_ch)
        return y, e, self.w


class FilterNLMS(AdaptiveFilter):
    def __init__(self, in_ch, out_ch, mu=0.1, eps=0.001, **kwargs):
        """
        Args:
            in_ch (int): number of input channels
            out_ch (int): number of output channels
            mu (float): learning rate
            eps (float): regularization term. It is introduced to preserve
                stability for close-to-zero input vectors
        """
        super().__init__(in_ch, out_ch, mu, **kwargs)
        self.eps = eps

    def learning_rule(self, e, x):
        """
        Args:
            e (ndarray (out_ch,)): error vector
            x (ndarray (in_ch,)): input vector
        Returns:
            w (ndarray (in_ch, out_ch)): updated weights
        """
        return self.mu / (self.eps + x @ x) * x[:, np.newaxis] @ e[np.newaxis, :]
