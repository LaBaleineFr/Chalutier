#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import sys
import time
import requests
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, wait


def get_ochl(currency):
    """ Get OCHL Data from Poloniex or Bittrex

    :param currency: Currency wanted
    :type currency: str
    :return: OCHL Data
    :rtype: DataFrame
    """
    end = round(time.time())
    cur = currency.upper()
    url = ''.join((
        'https://bittrex.com/Api/v2.0/pub/market/GetTicks?marketName=BTC-',
        cur,
        '&tickInterval=thirtyMin&_=',
        str(end)
    ))
    content = requests.get(url)
    data = content.json()
    if not data['success']:
        return cur, {'error': currency}
    df = pd.DataFrame.from_dict(data['result'])
    df.rename(
        columns={'C': 'close', 'H': 'high', 'L': 'low', 'O': 'open', 'T': 'date', 'V': 'volume'},
        inplace=True
    )
    df['date'] = pd.to_datetime(df['date'])
    # 5 day * 30 minutes -> 240 ticks
    df = df.tail(240)
    df.set_index(['date'], inplace=True)
    return cur, df

def returns(df):
    """Compute the returns from a period to the next

    :param df:  OCHL Data
    :type df: DataFrame
    :return:
    """
    df_returns = df.copy()
    df_returns.fillna(method='ffill', inplace=True)
    df_returns.fillna(method='bfill', inplace=True)
    df_returns[1:] = (df/df.shift(1)) - 1
    df_returns.ix[0, :] = 0
    return df_returns

def rand_weights(n):
    """ Initiate an array of random weigths summed to 1.0

    :param n: Array length
    :type n: int
    :return: Array of random weigths
    :rtype: ndarray of float

    TODO : Implementation of short selling :
    abs(weights) summed to 1.0
    """
    weights = np.random.rand(n)
    return weights / np.sum(weights)

def evaluate_portefolio(wei, returns_vec):
    """ Given a repartition, compute expected return and risk from a portefolio

    :param wei: Weights for each currency
    :type wei: ndarray of float
    :return: expected return and risk
    :rtype: (float, float)
    """
    p = np.asmatrix(np.mean(returns_vec, axis=1))
    w = np.asmatrix(wei)
    c = np.asmatrix(np.cov(returns_vec))
    mu = w * p.T
    sigma = np.sqrt(w * c * w.T)
    return mu, sigma

def markowitz_optimization(historical_statuses, evaluate=False):
    """ Construct efficient Markowitz Portefolio

    :param historical_statuses: 5 days OCHL of at least two currencies
    :param eval: evaluate 1000 random portefolios
    :returns: weights, means, stds, opt_mean, opt_std
    TODO : implement short selling (numeric instability w/ constraints)
    """
    nb_currencies = len(historical_statuses)
    lowest_size = np.min([i['close'].size for i in historical_statuses])
    returns_vec = [returns(singlecurrency)['close'].tail(lowest_size).values for singlecurrency in
                   historical_statuses]



    def optimal_portfolio():
        def con_sum(t):
            # Short ? -> np.sum(np.abs(t))-1
            return np.sum(t) - 1

        def con_no_short(t):
            # Short support ? add constraint for all weight > 0 to force non short
            # Minimizer instability with constraint_sum !!
            pass

        def quadra_risk_portefolio(ws):
            ws = np.asmatrix(ws)
            c = np.asmatrix(np.cov(returns_vec))
            return ws * c * ws.T

        cons = [{'type': 'eq', 'fun': con_sum}, ]
        # Short ? add no_short constraint  -> cons.append...
        res = minimize(
            quadra_risk_portefolio,
            rand_weights(nb_currencies),
            method='SLSQP',
            constraints=cons,
            options={'disp': False, 'ftol':1e-16,}
        )
        return res.x

    if evaluate:
        n_portfolios = 1000
    else:
        n_portfolios = 1
    means, stds = np.column_stack([
        evaluate_portefolio(rand_weights(nb_currencies), returns_vec)
        for _ in range(n_portfolios)
    ])

    #weights = opt2(returns_vec).flatten() #imal_portfolio()
    weights = optimal_portfolio()
    opt_mean, opt_std = evaluate_portefolio(weights, returns_vec)
    return weights, means, stds, opt_mean, opt_std

def optimiz(currencies, debug):
    currencies = sorted(currencies)
    if len(currencies) < 2 or len(currencies) > 10:
        return {"error": "2 to 10 currencies"}
    max_workers = 4 if sys.version_info[1] < 5 else None
    executor = ThreadPoolExecutor(max_workers)
    data = dict(future.result() for future in wait([executor.submit(get_ochl, cur) for cur in currencies]).done)
    data = [data[cur] for cur in currencies]
    errors = [x['error'] for x in data if 'error' in x]
    if errors:
        return {"error": "Currencies not found : " + str(errors)}
    weights, m, s, a, b = markowitz_optimization(data, debug)
    if debug:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.plot(s, m, 'o', markersize=1)
        plt.plot(b, a, 'or')
        fig.savefig("chalu.png")
    result = dict()
    for i, cur in enumerate(currencies):
        result[cur] = weights[i]
    return {"result": result}
