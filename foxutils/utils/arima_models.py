import numpy as np
import pandas as pd

from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from tqdm import notebook
from itertools import product
import matplotlib.pyplot as plt


# ARIMA Models


# Train many SARIMA models to find the best set of parameters
def optimize_SARIMA(yvals, parameters_list, d, D, s):
    """
        Return dataframe with parameters and corresponding AIC

        parameters_list - list with (p, q, P, Q) tuples
        d - integration order
        D - seasonal integration order
        s - length of season
    """

    results = []
    best_aic = float('inf')

    for param in notebook.tqdm(parameters_list):
        try:
            model = SARIMAX(yvals, order=(param[0], d, param[1]), seasonal_order=(param[2], D, param[3], s)).fit(
                disp=-1)
        except:
            continue

        aic = model.aic

        # Save best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    # Sort in ascending order, lower AIC is better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)

    return result_table


def get_optimized_sarima(yvals):
    # Set initial values and some bounds
    ps = range(0, 5)
    d = 1
    qs = range(0, 5)
    Ps = range(0, 5)
    D = 1
    Qs = range(0, 5)
    s = 5

    # Create a list with all possible combinations of parameters
    parameters = product(ps, qs, Ps, Qs)
    parameters_list = list(parameters)
    len(parameters_list)

    result_table = optimize_SARIMA(yvals, parameters_list, d, D, s)

    # Set parameters that give the lowest AIC (Akaike Information Criteria)
    p, q, P, Q = result_table.parameters[0]

    best_model = SARIMAX(yvals, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit(disp=-1)
    print(best_model.summary())
    return best_model


def get_stepwise_auto_arima(yvals):
    best_model = auto_arima(yvals,
                            start_p=1, start_q=1,
                            max_p=7, max_q=7,
                            d=1, max_d=7,
                            trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)

    print(best_model.summary())
    return best_model


def get_manual_arima(yvals, p=2, d=3, q=2):
    arima_model = ARIMA(yvals, order=(p, d, q))
    best_model = arima_model.fit()
    print(best_model.summary())

    #  If the model is good, its residuals should look like white noise.
    residuals = best_model.resid[1:]
    fig, ax = plt.subplots(1, 2)
    residuals.plot(title='Residuals', ax=ax[0])
    residuals.plot(title='Density', kind='kde', ax=ax[1])
    plt.show()

    # acf_res = plot_acf(residuals)
    # pacf_res = plot_pacf(residuals, method='ywm')

    return best_model


def reverse_diff(diff_vals, orig_first):
    return np.r_[orig_first, diff_vals].cumsum()


def forecast_values(model, pred_length):
    forecast_vals = model.forecast(steps=pred_length)
    # forecast_vals = best_model.predict(start=0, end=pred_length)
    return forecast_vals


def predict_values(model, pred_length):
    forecast_vals = model.predict(n_periods=pred_length)
    return forecast_vals
