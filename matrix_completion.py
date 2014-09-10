# Author: Vlad Niculae <vlad@vene.ro>
# Licence: BSD

from __future__ import division, print_function
import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.extmath import log_logistic
from sklearn.utils.fixes import expit

class SquaredLoss(object):
    def loss(self, y, pred):
        return 0.5 * (pred - y) ** 2

    def dloss(self, y, pred): 
        return pred - y

class LogLoss(object):
    def loss(self, y, pred):
        return (-log_logistic(y * pred))[0][0]

    def dloss(self, y, pred):
        return (-y * expit(-y * pred))

class EpsilonInsensitive(object):
    def __init__(self, eps):
        self.eps = eps

    def loss(self, y, pred):
        loss = np.abs(y - pred) - self.eps
        if loss < 0:
            return 0
        else:
            return loss

    def dloss(self, y, pred):
        raise NotImplemented


class HingeLoss(object):
    def loss(self, y, pred):
        loss = 1 - y * pred
        if loss < 0:
            return 0
        else:
            return loss

    def dloss(self, y, pred):
        z = pred * y
        if z <= 1:
            return -y
        else:
            return 0.0


class MatrixCompletion(object):
    """Online matrix completion by factorization.

    Estimates U, V to minimize loss(X, U * V.T).

    This is a slow, pure-python implementation.

    Basic support for classification when `method='pa'` and
    `is_classification=True`.

    Parameters
    ----------
    n_components : int, default: 2
        Number of components.

    method : {'sgd' | 'pa'}, default: 'pa'
        Optimization method:
            - 'sgd': stochastic gradient descent updates. Defaults to using
                     the squared loss for regression and the hinge loss
                     for classification.
            - 'pa': passive-aggressive updates. Supports regression (epsilon
                    insensitive loss) and classification (hinge loss). This
                    implements PA-I and the aggressiveness parameter is `alpha`.
                    Supports non-negative constraints only for regression.

    is_classification : boolean, default: False
        Whether to predict binary (+- 1) values.

    classification_loss: {'hingeloss' | 'logloss'}, default: 'hingeloss'
        Loss function to use for classification task. 

    non-negative : boolean, default: False
        Whether to enforce non-negativity constraints on the weights.  For
        `method=='sgd'` this just applies a projection step.  For `method=='pa'`
        this implements Blondel et al., Online Passive-Aggressive Algorithms for
        Non-Negative Matrix Factorization and Completion.

        As implemented, only makes sense for regression.

    U_init : array, shape=[n_samples, n_components] or None
        Initial points for the U factor.  If None, initialized to
        rand(-0.01, 0.01) + mean(X) / sqrt(n_components)

    V_init : array, shape=[n_features, n_components] or None
        Initial points for the V factor.  If None, initialized to
        rand(-0.01, 0.01) + mean(X) / sqrt(n_components)

    alpha : float, default: 0.001
        If `method='sgd'`, amount of L2 regularization. If `method='pa'`,
        aggressiveness factor.

    eps : float, default: 0.01
        Parameter of the epsilon-insensitive loss. Only used if `method='pa'`
        and `is_classification=False`.

    initial_learning_rate : float, default: 1e-5
        Learning rate to use at the first iteration.  Only used if
        `method='sgd'`.

    pow_t : float, default: 0.5
        If `method='sgd'`, the learning rate at iteration t is:
            initial_learning_rate / t ** pow_t

    tol : float, default: 1e-5
        Convergence criterion: absolute difference between consecutive
        cumulative losses should be less than it.

    max_iter : int, default: 1e5
        Maximum number of iterations to perform.

    shuffle : boolean, default: False
        Whether to shuffle the observations at each iteration.

    verbose : int, default: false
        Will print debug information at iterations divisible by it.

    random_state : int or RandomState
        Random number generator seed control.

    """
    def __init__(self, n_components=2, method='pa', is_classification=False, 
                 classification_loss='hingeloss', non_negative=False, U_init=None, 
                 V_init=None, alpha=0.001, eps=0.01, initial_learning_rate=1e-5, 
                 pow_t=0.5, tol=1e-5, max_iter=int(1e5), shuffle=False, verbose=False,
                 random_state=None):
        self.n_components = n_components
        self.method = method
        self.is_classification = is_classification
        self.classification_loss = classification_loss
        self.non_negative = non_negative
        self.U_init = U_init
        self.V_init = V_init
        self.alpha = alpha
        self.eps = eps
        self.initial_learning_rate = initial_learning_rate
        self.pow_t = pow_t
        self.tol = tol
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.verbose = verbose
        self.random_state = random_state

    def _init(self, X, mask, rng):
        n_samples, n_features = X.shape
        row_mask, col_mask = mask
        if self.is_classification:
            mean_x = 0
        else:
            mean_x = np.sqrt(X[row_mask, col_mask].mean() / self.n_components)

        if self.U_init is None:
            U = rng.uniform(-0.01, 0.01, size=(n_samples, self.n_components))
            U += mean_x
        else:
            U = self.U_init.copy()

        if self.V_init is None:
            V = rng.uniform(-0.01, 0.01, size=(n_features, self.n_components))
            V += mean_x
        else:
            V = self.V_init.copy()

        if self.non_negative:
            # make sure we start in the feasible region
            np.clip(U, 0, np.inf, out=U)
            np.clip(V, 0, np.inf, out=V)
        return U, V

    def _update(self, w, x, y, pred=None):
        if self.method == 'pa':
            return self._pa_update(w, x, y, pred)
        else:
            return self._sgd_update(w, x, y, pred)

    def _sgd_update(self, w, x, y, pred):
        """Squared loss gradient update"""
        return -(self.loss_.dloss(y, pred) * x + self.alpha * w)

    def _pa_update(self, w, x, y, pred):
        # for notation
        C = self.alpha

        loss = self.loss_.loss(y, pred)  # hinge or eps-insensitive

        if loss == 0:
            return 0

        loss /= np.sum(x ** 2)
        if loss > C:
            loss = C

        if self.is_classification:
            loss *= y
        elif y < pred:
            if self.non_negative:
                f_C = np.dot(np.clip(w - C * x, 0, np.inf), x) - y - self.eps
                if f_C >= 0:
                    loss = C
            loss *= -1

        return loss * x

    def fit(self, X, y=None, mask=None):
        rng = check_random_state(self.random_state)

        if not mask:
            row_mask, col_mask = X.nonzero()
        else:
            row_mask, col_mask = mask

        # initialize as pyrsvd, so that U_init * V_init.T == X[mask].mean()
        U, V = self._init(X, (row_mask, col_mask), rng)

        n_nonzero = len(row_mask)
        indices = np.arange(n_nonzero)

        if 'pa' in self.method:
            if self.is_classification:
                if 'hingeloss' in self.classification_loss: 
                    self.loss_ = HingeLoss()
                else:
                    self.loss_ = LogLoss() # this works too right?
                if self.non_negative:
                    raise NotImplementedError("Non-negativity constraints for "
                                              "passive-aggressive with hinge "
                                              "loss are not implemented "
                                              "(do they even make sense?)")
            else:
                self.loss_ = EpsilonInsensitive(eps=self.eps)
        else:
            if self.is_classification:
                if 'hingeloss' in self.classification_loss:
                    self.loss_ = HingeLoss()
                else:
                    self.loss_ = LogLoss()
            else:
                self.loss_ = SquaredLoss()

        old_cumulative_loss = np.inf

        numpy_error_settings = np.seterr(all='raise')
        for ii in range(self.max_iter):
            if 'pa' in self.method:
                learning_rate = 1.0
            else:
                # inverse scaling learning rate, as in scikit-learn SGDRegressor
                learning_rate = self.initial_learning_rate / ((1 + ii) **
                                                              self.pow_t)
            cumulative_loss = 0
            if self.shuffle:
                rng.shuffle(indices)

            for idx in indices:
                row, col = row_mask[idx], col_mask[idx]

                pred = np.dot(U[row], V[col])
                y = X[row, col]
                # Keep track of sum of squared errors for all data
                loss = self.loss_.loss(y, pred)

                cumulative_loss += loss

                u_update = self._update(U[row], V[col], y, pred)
                v_update = self._update(V[col], U[row], y, pred)

                U[row] += learning_rate * u_update
                V[col] += learning_rate * v_update

                if self.non_negative:
                    np.clip(U[row], 0, np.inf, out=U[row])
                    np.clip(V[col], 0, np.inf, out=V[col])

            if np.abs(old_cumulative_loss - cumulative_loss) < self.tol:
                if self.verbose:
                    print("Converged")
                break

            if self.verbose and not ii % self.verbose:
                print("Iteration", ii, ", loss:", cumulative_loss)

            old_cumulative_loss = cumulative_loss
        np.seterr(**numpy_error_settings)
        if self.verbose:
            print("Iteration", ii, ", loss:", cumulative_loss)

        self.U_, self.V_ = U, V
        return self


def example_regression():
    rng = np.random.RandomState(0)
    U_true = rng.rand(50, 2) * 2
    V_true = rng.rand(40, 2) * 2
    X = np.dot(U_true, V_true.T)
    print(X[:5, :5])

    # Put aside 1500 entries for testing
    row_mask = rng.randint(0, 50, size=1500)
    col_mask = rng.randint(0, 40, size=1500)

    # Get a mask for the remaining (training) observations
    mask = np.zeros((50, 40), dtype=np.bool)
    mask[row_mask, col_mask] = 1
    fit_mask = (~mask).nonzero()

    MF = MatrixCompletion(method='pa', n_components=2, non_negative=True,
                          random_state=0, alpha=0.1, eps=0.01, verbose=1,
                          shuffle=True)
    MF.fit(X, mask=fit_mask)
    U, V = MF.U_, MF.V_
    X_pred = np.dot(U, V.T)
    test_err = np.sum((X_pred[row_mask, col_mask] - X[row_mask, col_mask]) ** 2)
    print("Test resid. norm ", np.sqrt(test_err))
    print(np.dot(U, V.T)[:5, :5])


def example_classification(method='pa', classification_loss='hingeloss', initial_learning_rate=1e-4,
                            tol=0.01):
    noise = 0.1
    rng = np.random.RandomState(0)
    U_true = rng.randn(50, 2)
    V_true = rng.randn(40, 2)
    X = np.sign(np.dot(U_true, V_true.T) + noise * rng.randn(50, 40))
    print(X[:5, :5])

    # Put aside 1500 entries for testing
    row_mask = rng.randint(0, 50, size=1500)
    col_mask = rng.randint(0, 40, size=1500)
    # Get a mask for the remaining (training) observations
    mask = np.zeros((50, 40), dtype=np.bool)
    mask[row_mask, col_mask] = 1
    fit_mask = (~mask).nonzero()

    MF = MatrixCompletion(method=method, is_classification=True, 
                          classification_loss = classification_loss,
                          initial_learning_rate=initial_learning_rate,
                          n_components=2,random_state=0, alpha=0.1,
                          verbose=10, shuffle=True, tol=tol)

    MF.fit(X, mask=fit_mask)

    U, V = MF.U_, MF.V_
    X_pred = np.dot(U, V.T)
    test_acc = np.mean(np.sign(X_pred[row_mask, col_mask]) ==
                       X[row_mask, col_mask])

    print("Test accuracy:", test_acc)
    print(np.dot(U, V.T)[:5, :5])


if __name__ == '__main__':
    example_classification()
