#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
import requests
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED


def get_ochl(currency):
    end = round(time.time())
    # 5 days of data, 30 min periods
    start = end - 5 * 86400
    cur = currency.upper()

    def poloniex():
        url = "https://poloniex.com/public?command=returnChartData&currencyPair=BTC_" + cur + "&start=" + str(start) + \
              "&end=" + str(end) + "&period=1800"
        content = requests.get(url)
        data = content.json()
        if 'error' in data:
            return {'error': "Currency not found : " + currency}
        df = pd.DataFrame.from_dict(data)
        df['date'] = pd.to_datetime(df['date'], unit='s')
        df.set_index(['date'], inplace=True)
        return df

    def bittrex():
        url = "https://bittrex.com/Api/v2.0/pub/market/GetTicks?marketName=BTC-" + cur + "&tickInterval=thirtyMin&_=" +\
              str(end)
        content = requests.get(url)
        data = content.json()
        if not data['success']:
            return {'error': "Currency not found : " + currency}
        df = pd.DataFrame.from_dict(data['result'])
        df.rename(columns={'C': 'close', 'H': 'high', 'L': 'low', 'O': 'open', 'T': 'date', 'V': 'volume'},
                  inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        # keep consistent between polo and bittrex 5 day * 30 minutes -> 240 ticks
        df = df.tail(240)
        df.set_index(['date'], inplace=True)
        return df

    executor = ThreadPoolExecutor()
    future = wait([executor.submit(poloniex), executor.submit(bittrex)], return_when=FIRST_COMPLETED)
    df = future.done.pop().result()

    return df if 'error' not in df else future.not_done.pop().result()

def returns(df):
    """
    Compute the returns from a period to the next
    :param df:  Lows, Highs, Opens, Closes
    :return:
    """
    df_returns = df.copy()
    df_returns.fillna(method='ffill', inplace=True)
    df_returns.fillna(method='bfill', inplace=True)
    df_returns[1:] = (df/df.shift(1)) - 1
    df_returns.ix[0, :] = 0
    return df_returns

def markowitz_optimization(historicalstatuses, eval=False):
    """
    :param historicalstatuses: 5 days OCHL of at least two currencies
    :param eval: evaluate 1000 random portefolios
    :returns: weights, means, stds, opt_mean, opt_std
    # TODO implement short selling (numeric instability w/ constraints)
    """

    lowest_index = np.min([i['close'].size for i in historicalstatuses])
    returns_vec = [returns(singlecurrency)['close'].ix[:lowest_index - 1].values for singlecurrency in
                   historicalstatuses]

    def rand_weights():
        # Short ?
        weights = np.random.rand(len(historicalstatuses))
        return weights / np.sum(weights)

    def evaluate_portefolio(wei):
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

        wei = rand_weights()
        cons = [{'type': 'eq', 'fun': con_sum}, ]
        # Short ? add no_short constraint  -> cons.append...
        res = minimize(quadra_risk_portefolio, wei, constraints=cons, tol=1e-10, options={'disp': False})
        return res.x

    if eval:
        n_portfolios = 1000
    else:
        n_portfolios = 1
    means, stds = np.column_stack([
        evaluate_portefolio(rand_weights())
        for _ in range(n_portfolios)
    ])

    weights = optimal_portfolio()
    opt_mean, opt_std = evaluate_portefolio(weights)
    return weights, means, stds, opt_mean, opt_std

def optimiz(currencies, debug):
    if len(currencies) < 2 or len(currencies) > 10:
        return {"error": "2 to 10 currencies"}
    executor = ThreadPoolExecutor()
    data = [future.result() for future in wait([executor.submit(get_ochl, cur) for cur in currencies]).done]
    errors = [x['error'] for x in data if 'error' in x]
    if errors:
        return {"error": "\n".join(errors)}
    weights, m, s, a, b = markowitz_optimization(data, debug)
    if debug is True:
        fig, ax = plt.subplots()
        plt.plot(s, m, 'o', markersize=1)
        plt.plot(b, a, 'or')
        fig.savefig("chalu.png")
    result = dict()
    for i, cur in enumerate(currencies):
        result[cur] = weights[i]
    return {"result": result}
