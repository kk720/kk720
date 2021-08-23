import numpy as np
import pandas as pd
import requests
import math
from scipy import stats

stocks = pd.read_csv('sp_500_stocks.csv')


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


symbol_groups = list(chunks(stocks['Ticker'], 100))
symbol_strings = []
for i in range(0, len(symbol_groups)):
    symbol_strings.append(','.join(symbol_groups[i]))
rv_columns = ['Ticker',
              'Price',
              'PE ratio',
              'PE percentile',
              'PB ratio',
              'PB percentile',
              'PS ratio',
              'PS percentile',
              'EV/EBITDA',
              'EV/EBITDA percentile',
              'EV/GP',
              'EV/GP percentile',
              'RV score',
              'Number of Shares to Buy']
rv_dataframe = pd.DataFrame(columns=rv_columns)
for symbol_string in symbol_strings:
    batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch/?types=quote,advanced-stats&symbols={symbol_string}&token=Tpk_6fbd75672968406baab167d063f13a9e'
    data = requests.get(batch_api_call_url).json()
    for symbol in symbol_string.split(','):
        enterprise_value = data[symbol]['advanced-stats']['enterpriseValue']
        ebitda = data[symbol]['advanced-stats']['EBITDA']
        gross_profit = data[symbol]['advanced-stats']['grossProfit']
        try:
            ev_to_ebitda = enterprise_value/ebitda
        except TypeError:
            ev_to_ebitda = np.NaN
        try:
            ev_to_gross_profit = enterprise_value/gross_profit
        except TypeError:
            ev_to_gross_profit = np.NaN
        rv_dataframe = rv_dataframe.append(
            pd.Series([symbol,
                       data[symbol]['quote']['latestPrice'],
                       data[symbol]['quote']['peRatio'],
                       'N/A',
                       data[symbol]['advanced-stats']['priceToBook'],
                       'N/A',
                       data[symbol]['advanced-stats']['priceToSales'],
                       'N/A',
                       ev_to_ebitda,
                       'N/A',
                       ev_to_gross_profit,
                       'N/A',
                       'N/A',
                       'N/A'
                       ],
                      index=rv_columns),
            ignore_index=True)
for column in ['Price', 'PE ratio', 'PB ratio', 'PS ratio', 'EV/EBITDA', 'EV/GP']:
    rv_dataframe[column].fillna(rv_dataframe[column].mean(), inplace=True)
metrics = {
              'PE ratio': 'PE percentile',
              'PB ratio': 'PB percentile',
              'PS ratio': 'PS percentile',
              'EV/EBITDA': 'EV/EBITDA percentile',
              'EV/GP': 'EV/GP percentile'
}
for metric in metrics.keys():
    for row in rv_dataframe.index:
        rv_dataframe.loc[row, metrics[metric]] = stats.percentileofscore(rv_dataframe[metric], rv_dataframe.loc[row, metric])

import statistics

for row in rv_dataframe.index:
    value_percentile = []
    for metric in metrics.keys():
        value_percentile.append(rv_dataframe.loc[row, metrics[metric]])
    rv_dataframe.loc[row, 'RV score'] = statistics.mean(value_percentile)

rv_dataframe.sort_values('RV score', inplace=True, ascending=True)
rv_dataframe = rv_dataframe[:50]
rv_dataframe.reset_index(drop=True, inplace=True)
print(rv_dataframe)

position_size = 50000000
for row in rv_dataframe.index:
    rv_dataframe.loc[row, 'Number of Shares to Buy'] = math.floor(position_size/rv_dataframe.loc[row, 'Price'])

rv_dataframe.to_excel('value stock trading preference.xlsx')




# final_dataframe = pd.DataFrame(columns=my_columns)
# for symbol_string in symbol_strings:
#     batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch/?types=quote&symbols={symbol_string}&token=Tpk_6fbd75672968406baab167d063f13a9e'
#     data = requests.get(batch_api_call_url).json()
#     for symbol in symbol_string.split(','):
#         final_dataframe = final_dataframe.append(
#             pd.Series([symbol,
#                        data[symbol]['quote']['latestPrice'],
#                        data[symbol]['quote']['peRatio'],
#                        'N/A'
#                        ],
#                       index=my_columns),
#             ignore_index=True)
# final_dataframe.sort_values('PE ratio', inplace=True, ascending=True)
# final_dataframe = final_dataframe[final_dataframe['PE ratio'] > 0]
# final_dataframe = final_dataframe[:50]
# final_dataframe.reset_index(inplace=True)
# final_dataframe.drop('index', axis=1, inplace=True)
# print(final_dataframe)
