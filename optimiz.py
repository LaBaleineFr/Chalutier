#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import sys
import time
import requests
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED


def get_ochl(currency, max_workers):
    """ Get OCHL Data from Poloniex or Bittrex

    :param currency: Currency wanted
    :param max_workers: Needed for Python < 3.5
    :type currency: str
    :type max_workers: int
    :return: OCHL Data
    :rtype: DataFrame
    """
    end = round(time.time())
    # 5 days of data, 30 min periods
    start = end - 5 * 86400
    cur = currency.upper()

    def poloniex():
        """ Pull data from Poloniex exchange """
        url = ''.join((
            'https://poloniex.com/public?command=returnChartData&currencyPair=BTC_',
            cur,
            '&start=',
            str(start),
            '&end=',
            str(end),
            '&period=1800'
        ))
        content = requests.get(url)
        data = content.json()
        if 'error' in data:
            return {'error': 'Currency not found : ' + currency}
        df = pd.DataFrame.from_dict(data)
        df['date'] = pd.to_datetime(df['date'], unit='s')
        df.set_index(['date'], inplace=True)
        return df

    def bittrex():
        """ Pull data from Bittrex exchange """
        url = ''.join((
            'https://bittrex.com/Api/v2.0/pub/market/GetTicks?marketName=BTC-',
            cur,
            '&tickInterval=thirtyMin&_=',
            str(end)
        ))
        content = requests.get(url)
        data = content.json()
        if not data['success']:
            return {'error': 'Currency not found : ' + currency}
        df = pd.DataFrame.from_dict(data['result'])
        df.rename(
            columns={'C': 'close', 'H': 'high', 'L': 'low', 'O': 'open', 'T': 'date', 'V': 'volume'},
            inplace=True
        )
        df['date'] = pd.to_datetime(df['date'])
        # keep consistent between polo and bittrex 5 day * 30 minutes -> 240 ticks
        df = df.tail(240)
        df.set_index(['date'], inplace=True)
        return df

    executor = ThreadPoolExecutor(max_workers=max_workers)
    future = wait(
        [executor.submit(poloniex), executor.submit(bittrex)],
        return_when=FIRST_COMPLETED
    )
    df = future.done.pop().result()

    return df if 'error' not in df else future.not_done.pop().result()

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

def markowitz_optimization(historical_statuses, eval=False):
    """ Construct efficient Markowitz Portefolio

    :param historical_statuses: 5 days OCHL of at least two currencies
    :param eval: evaluate 1000 random portefolios
    :returns: weights, means, stds, opt_mean, opt_std
    TODO : implement short selling (numeric instability w/ constraints)
    """
    nb_currencies = len(historical_statuses)
    lowest_index = np.min([i['close'].size for i in historical_statuses])
    returns_vec = [returns(singlecurrency)['close'].ix[:lowest_index - 1].values for singlecurrency in
                   historical_statuses]

    def evaluate_portefolio(wei):
        """ Given a repartition, compute the expected return and risk from a portefolio 
        
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

        wei = rand_weights(nb_currencies)
        cons = [{'type': 'eq', 'fun': con_sum}, ]
        # Short ? add no_short constraint  -> cons.append...
        res = minimize(quadra_risk_portefolio, wei, constraints=cons, tol=1e-10, options={'disp': False})
        return res.x

    if eval:
        n_portfolios = 1000
    else:
        n_portfolios = 1
    means, stds = np.column_stack([
        evaluate_portefolio(rand_weights(nb_currencies))
        for _ in range(n_portfolios)
    ])

    weights = optimal_portfolio()
    opt_mean, opt_std = evaluate_portefolio(weights)
    return weights, means, stds, opt_mean, opt_std

def optimiz(currencies, debug):
    if len(currencies) < 2 or len(currencies) > 10:
        return {"error": "2 to 10 currencies"}
    max_workers = 4 if sys.version_info[1] < 5 else None
    executor = ThreadPoolExecutor(max_workers)
    data = [future.result() for future in wait([executor.submit(get_ochl, cur, max_workers) for cur in currencies]).done]
    errors = [x['error'] for x in data if 'error' in x]
    if errors:
        return {"error": "\n".join(errors)}
    weights, m, s, a, b = markowitz_optimization(data, debug)
    if debug:
        fig, ax = plt.subplots()
        plt.plot(s, m, 'o', markersize=1)
        plt.plot(b, a, 'or')
        fig.savefig("chalu.png")
    result = dict()
    for i, cur in enumerate(currencies):
        result[cur] = weights[i]
    return {"result": result}
