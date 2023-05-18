from abc import ABC, abstractmethod
from pprint import pprint

import numpy as np
import pandas as pd
from pandas import json_normalize


class Processor(ABC):
    @abstractmethod
    def process(self, data):
        pass


# Todo Fix . from the json normalize function in columns names
class GlassNodeProcessor(Processor):
    """
    Processor for GlassNode API data
    """

    def process(self, data, metric_name):  # noqa
        """
        Process data for Glassnode endpoints
        :param data: json response
        :param metric_name: name of metric
        :return: pandas dataframe
        """
        df = json_normalize(data)
        df.set_index('t', inplace=True)
        if len(df.columns) > 1:
            df.columns = [x.split('.')[1] for x in df.columns]
        else:
            df.columns = [metric_name]
        df.index = pd.to_datetime(df.index, unit='s')
        return df


class CoinGeckoProcessor(Processor):
    """

    Processor for CoinGecko API data
    """

    def process(self, data, metric_name):  # noqa
        """
        Process data for CoinGecko endpoints
        :param data: json response
        :param metric_name: name of metric
        :return: pandas dataframe
        """
        col_names = ['Price', 'Market Cap']

        # Subsets the data to only include the price and market cap
        prices = pd.DataFrame(data['prices'])
        prices.set_index(0, inplace=True)
        prices.columns = [col_names[0]]

        mcap = pd.DataFrame(data['market_caps'])
        mcap.set_index(0, inplace=True)
        mcap.columns = [col_names[1]]

        df = prices.join(mcap, how='outer')
        df.index = pd.to_datetime(df.index, unit='ms')
        df = df.resample('D').last()
        return df


class YahooFinanceProcessor(Processor):
    """
    Processor for Yahoo Finance API data
    """

    def process(self, data, metric_name):  # noqa
        """
        Process data for Yahoo Finance endpoints
        :param data: json response
        :param metric_name: name of metric
        :return: pandas dataframe
        """
        adjusted_close_values = data['chart']['result'][0]['indicators']['adjclose'][0]['adjclose']
        timestamps = data['chart']['result'][0]['timestamp']
        df = pd.DataFrame(adjusted_close_values, index=timestamps, columns=[metric_name])
        df.index = pd.to_datetime(df.index, unit='s')
        return df.resample('D').last()


class FredProcessor(Processor):
    """
    Processor for Fred API data
    """

    def process(self, data, metric_name):  # noqa
        """
        Process data for Fred endpoints
        :param data: json response
        :param metric_name: name of metric
        :return: pandas dataframe
        """
        df = pd.DataFrame(data['observations'])
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
        df = df[['value']]
        df.columns = [metric_name]
        df = df.replace('.', np.nan)
        df = df.astype(float)
        return df


class CoinGlassProcessor(Processor):
    """
    Processor for CoinGlass API data
    """

    def process(self, data, metric_name):  # noqa
        """
        Process data for CoinGlass endpoints
        :param data: json response
        :param metric_name: name of metric
        :return: pandas dataframe
        """
        timestamps = data['data']['dateList']
        datamap = data['data']['dataMap']
        df = pd.DataFrame.from_dict(datamap, orient='columns')
        df.index = pd.to_datetime(timestamps, unit='ms')
        df['Aggregated'] = df.sum(axis=1)
        return df


