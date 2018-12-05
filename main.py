import numpy as np
import sys
import datetime
import GPy
import itertools

from scipy.optimize import minimize
import matplotlib.pyplot as plt

from osprey.config import Config
from osprey.trials import Trial
from osprey.strategies import GP

def from_gp(result, searchspace):

    # Note that GP only deals with float-valued variables, so we have
    # a transform step on either side, where int and enum valued variables
    # are transformed before calling gp, and then the result suggested by
    # GP needs to be reverse-transformed.
    out = {}
    for gpvalue, var in zip(result, searchspace):
        out[var.name] = var.point_from_gp(float(gpvalue))

    return out

def get_data(history, searchspace):
    """
    Get features, mean scores, variance of scores and elapsed time in seconds
    """
    X = []
    Y = []
    V = []
    TY = []
    ignore = []
    all_scores = []
    for param_dict, scores, status, elapsed in history:
        # transform points into the GP domain. This invloves bringing
        # int and enum variables to floating point, etc.
        if status == 'FAILED':
            # not sure how to deal with these yet
            continue

        point = searchspace.point_to_gp(param_dict)
        if status == 'SUCCEEDED':
            X.append(point)
            all_scores.append(scores)
            TY.append(elapsed.total_seconds())

        elif status == 'PENDING':
            ignore.append(point)
        else:
            raise RuntimeError('unrecognized status: %s' % status)

    transformed_scores = np.log(- np.asarray(all_scores))
    transformed_time = np.log(TY)
    transform_params = (np.mean(transformed_scores), np.std(transformed_scores, ddof=1),
                        np.mean(transformed_time), np.std(transformed_time, ddof=1))
    transformed_scores -= transform_params[0]
    transformed_scores /= transform_params[1]
    transformed_time -= transform_params[2]
    transformed_time /= transform_params[3]


    # Include all cv folds in the scoring instead of using averages
    Y = transformed_scores.ravel()
    TY = list(itertools.chain.from_iterable(itertools.repeat(x,3) for x in transformed_time))
    X = list(itertools.chain.from_iterable(itertools.repeat(x,3) for x in X))

    return (np.array(X).reshape(-1, searchspace.n_dims),
            np.array(Y).reshape(-1, 1),
            np.array(TY).reshape(-1,1),
            transform_params)

def data_from_config(config_file):
    """
    Returns features, scores, score variance, elapsed time in seconds
    and searchspace from a config file
    """
    config = Config(config_file)
    session = config.trials()

    searchspace = config.search_space()
    history = [[t.parameters, t.test_scores, t.status, t.elapsed]
                       for t in session.query(Trial).all()]

    return get_data(history, searchspace) + (searchspace,)

def score_mae(y, y_pred):
    return np.sum(abs(y-y_pred)) / y.size

def fit_gp(X, Y):
    """
    Fit GP
    """
    kernels =  [GPy.kern.RBF(input_dim=X.shape[1],ARD=True) + GPy.kern.Bias(X.shape[1]),
                GPy.kern.RBF(input_dim=X.shape[1],ARD=True),
                GPy.kern.Matern52(input_dim=X.shape[1],ARD=True) + GPy.kern.Bias(X.shape[1]),
                GPy.kern.Matern52(input_dim=X.shape[1],ARD=True),
                GPy.kern.Matern32(input_dim=X.shape[1],ARD=True) + GPy.kern.Bias(X.shape[1]),
                GPy.kern.Matern32(input_dim=X.shape[1],ARD=True),
                GPy.kern.Exponential(input_dim=X.shape[1],ARD=True) + GPy.kern.Bias(X.shape[1]),
                GPy.kern.Exponential(input_dim=X.shape[1],ARD=True)]
    models = []
    for kernel in kernels:

        #if V is not None:
        #    kernel += GPy.kern.WhiteHeteroscedastic(X.shape[1],X.shape[0])

        model = GPy.models.GPRegression(X,Y,kernel)

        #if V is not None:
        #    model.sum.white_hetero.variance.fix(V.ravel())

        model.optimize_restarts(num_restarts=5, robust=True, verbose=False)
        models.append(model)

    aicc = []
    for i, model in enumerate(models):
        aicc.append(ic(model, 'aicc'))

    return models[np.argmin(aicc)]

def ic(model,method='bic'):
    ll = model.log_likelihood()
    n = model.num_data
    k = model.optimizer_array.size

    if method == 'bic':
        return np.log(n) * k - 2 * ll
    elif method == 'aicc':
        return 2 * k - 2 * ll + (2 * k**2 + 2 * k) / (n - k - 1)

def get_gp_best(score_model, time_model, transform_params, freeze_idx=None, freeze_value=None,
        nsamples = 10000, nopt = 100, time_limit = 36):
    # Objective function
    def z(x):
        if freeze_idx is not None:
            xold = x.copy()
            x = np.zeros(xold.size + 1)
            x[:freeze_idx] = xold[:freeze_idx]
            x[freeze_idx] = freeze_value
            x[freeze_idx+1:] = xold[freeze_idx:]
            del xold

        y_mean, y_var = score_model.predict(x.reshape(1,-1))

        # Return score + 1 stdev to select regions we are confident are good
        return float(y_mean + np.sqrt(y_var))

    # time constraint
    def con(x):
        if freeze_idx is not None:
            xold = x.copy()
            x = np.zeros(xold.size + 1)
            x[:freeze_idx] = xold[:freeze_idx]
            x[freeze_idx] = freeze_value
            x[freeze_idx+1:] = xold[freeze_idx:]
            del xold

        y_mean, y_var = time_model.predict(x.reshape(1,-1))

        actual_time = np.exp(float(y_mean) * transform_params[3] + transform_params[2])
        # Constrain time spent to 10 hours (3-fold CV)
        return time_limit - actual_time / 3600

    nfeatures = score_model.input_dim

    if freeze_idx is not None:
        nfeatures -= 1

    # Generate random points
    X = np.random.random((nsamples, nfeatures))

    # Predict and keep only top 100
    if freeze_idx is None:
        Y, _ = score_model.predict(X)
        Z, _ = time_model.predict(X)
    else:
        X2 = np.zeros((X.shape[0], X.shape[1] + 1))
        X2[:,:freeze_idx] = X[:,:freeze_idx]
        X2[:,freeze_idx] = freeze_value
        X2[:,freeze_idx+1:] = X[:,freeze_idx:]
        Y, _ = score_model.predict(X2)
        Z, _ = time_model.predict(X2)

    time_mask = np.exp(Z.ravel() * transform_params[3] + transform_params[2]) / 3600 < time_limit

    idx = np.argpartition(Y.ravel()[time_mask], nopt)[:nopt]
    X = X[time_mask][idx]

    # Optimize using these points at starting positions
    topx, topy = [], []
    for x in X:
        res = minimize(z, x, bounds=nfeatures*[(0., 1.)],
                        constraints=[{'type': 'ineq', 'fun': con}],
                        options={'maxiter': 1000, 'disp': 0})
        topx.append(res.x)
        topy.append(res.fun)

    best_idx = np.argmin(topy)
    if freeze_idx is None:
        x = topx[best_idx]
    else:
        xbest = topx[best_idx]
        x = np.zeros(nfeatures + 1)
        x[:freeze_idx] = xbest[:freeze_idx]
        x[freeze_idx] = freeze_value
        x[freeze_idx+1:] = xbest[freeze_idx:]

    return x, topy[best_idx]

def plot_single(score_model, time_model, transform_params):

    nfeatures = score_model.input_dim


    for i in range(nfeatures):
        X, Y = [], []
        rang = np.linspace(0,1,20)
        for value in rang:
            x, y = get_gp_best(score_model, time_model, transform_params,
                    freeze_idx=i, freeze_value=value, nsamples = 100, nopt = 10)

            y = np.exp(float(y)*transform_params[1] + transform_params[0])
            x = list(searchspace.point_from_gp(x).values())[i]
            X.append(x)
            Y.append(y)

        xname = list(searchspace.point_from_gp(np.zeros(nfeatures)).keys())[i]

        with open("%s.txt" % xname, "w") as f:
            for j, x in enumerate(X):
                f.write("%s %s" % (str(x), str(Y[j])))

        plt.title(xname)
        plt.loglog(X,Y)
        plt.savefig("%s.png" % xname)
        plt.clf()


if __name__ == "__main__":
    config_file = sys.argv[1]
    X, Y, TY, transform_params, searchspace = data_from_config(config_file)
    score_model = fit_gp(X,Y)
    time_model = fit_gp(X,TY)


    x, y = get_gp_best(score_model, time_model, transform_params,
            nsamples=1000, nopt=100, time_limit=24)
    y = np.exp(float(y)*transform_params[1] + transform_params[0])
    x = searchspace.point_from_gp(x)
    print(y)
    print(x)
    quit()
    plot_single(score_model, time_model, transform_params)
