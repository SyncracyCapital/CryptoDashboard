import asyncio
from datetime import datetime, timedelta
import time

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import merge_dataframes
from metrics import Metric
from processors import GlassNodeProcessor, CoinGeckoProcessor, YahooFinanceProcessor, FredProcessor, CoinGlassProcessor
from utils import prepare_data

import streamlit as st

# App configuration
st.set_page_config(
    page_title="Data Catalog",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

# App Header
markdown = """<h1 style='font-family: Calibri; text-align: center;'><img 
src="https://images.squarespace-cdn.com/content/v1/63857484f91d71181b02f971/9943adcc-5e69-489f-b4a8-158f20fe0619
/Snycracy_WebLogo.png?format=700w" alt="logo"/></h1>"""

st.markdown(markdown, unsafe_allow_html=True)

st.markdown('-------------------')

# Todo Move this to secrets.toml
# Data Sources configuration
# Glassnode settings
base_api_url = 'https://api.glassnode.com/v1/metrics'
api_key = st.secrets['GLASSNODE_API_KEY']
glassnode_processor = GlassNodeProcessor()

# CoinGecko settings
base_api_cg = 'https://pro-api.coingecko.com/api/v3/coins'
api_key_cg = st.secrets['COINGECKO_API_KEY']
cg_processor = CoinGeckoProcessor()

# Yahoo Finance settings
base_api_yf = 'https://query1.finance.yahoo.com/v8/finance'
yf_processor = YahooFinanceProcessor()
today = datetime.today()
unix_today = int((time.mktime(today.timetuple())))

# FRED settings
base_api_fred = 'https://api.stlouisfed.org/fred/series'
api_key_fred = st.secrets['FRED_API_KEY']
fred_processor = FredProcessor()

# Coinglass settings
base_api_coinglass = 'https://open-api.coinglass.com/public/v2'
api_key_coinglass = st.secrets['COINGLASS_API_KEY']
coinglass_processor = CoinGlassProcessor()

# Array of Metrics objects to fetch
metrics = [
    # Glassnode Metrics
    Metric(base_api_url, 'market', api_key, metric_name='mvrv', params={'a': 'BTC', 'api_key': api_key},
           processor=glassnode_processor, asset_name='BTC', df_col_name='BTC MVRV'),
    Metric(base_api_url, 'supply', api_key, metric_name='rcap_hodl_waves', params={'a': 'BTC', 'api_key': api_key},
           processor=glassnode_processor, asset_name='BTC', df_col_name='BTC Realized Cap HODL Waves'),
    Metric(base_api_url, 'supply', api_key, metric_name='lth_sum_pit', params={'a': 'BTC', 'api_key': api_key},
           processor=glassnode_processor, asset_name='BTC', df_col_name='BTC LTH Supply'),
    Metric(base_api_url, 'supply', api_key, metric_name='sth_sum_pit', params={'a': 'BTC', 'api_key': api_key},
           processor=glassnode_processor, asset_name='BTC', df_col_name='BTC STH Supply'),
    Metric(base_api_url, 'distribution', api_key, metric_name='exchange_net_position_change',
           params={'a': 'BTC', 'api_key': api_key}, processor=glassnode_processor, asset_name='BTC',
           df_col_name='BTC Exchange Net Position Change'),
    Metric(base_api_url, 'distribution', api_key, metric_name='exchange_net_position_change',
           params={'a': 'ETH', 'api_key': api_key}, processor=glassnode_processor, asset_name='ETH',
           df_col_name='ETH Exchange Net Position Change'),
    Metric(base_api_url, 'fees', api_key, metric_name='gas_price_mean', params={'a': 'ETH', 'api_key': api_key},
           processor=glassnode_processor, asset_name='ETH', df_col_name='ETH Gas Price Mean'),
    Metric(base_api_url, 'transactions', api_key, metric_name='count', params={'a': 'ETH', 'api_key': api_key},
           processor=glassnode_processor, asset_name='ETH', df_col_name='ETH Transactions Count'),
    Metric(base_api_url, 'fees', api_key, metric_name='tx_types_breakdown_relative',
           params={'a': 'ETH', 'api_key': api_key},
           processor=glassnode_processor, asset_name='ETH', df_col_name='ETH Transaction Types Breakdown'),
    Metric(base_api_url, 'addresses', api_key, metric_name='active_count', params={'a': 'ETH', 'api_key': api_key},
           processor=glassnode_processor, asset_name='ETH', df_col_name='ETH Active Addresses Count'),

    Metric(base_api_url, 'transactions', api_key, metric_name='transfers_volume_sum',
           params={'a': 'USDT', 'api_key': api_key},
           processor=glassnode_processor, asset_name='USDT', df_col_name='USDT Transfers Volume'),
    Metric(base_api_url, 'transactions', api_key, metric_name='transfers_volume_sum',
           params={'a': 'USDT', 'api_key': api_key},
           processor=glassnode_processor, asset_name='USDT', df_col_name='USDC Transfers Volume'),
    Metric(base_api_url, 'transactions', api_key, metric_name='transfers_volume_sum',
           params={'a': 'BUSD', 'api_key': api_key},
           processor=glassnode_processor, asset_name='BUSD', df_col_name='BUSD Transfers Volume'),
    Metric(base_api_url, 'transactions', api_key, metric_name='transfers_volume_sum',
           params={'a': 'DAI', 'api_key': api_key},
           processor=glassnode_processor, asset_name='DAI', df_col_name='DAI Transfers Volume'),

    Metric(base_api_url, 'market', api_key, metric_name='marketcap_realized_usd',
           params={'a': 'BTC', 'api_key': api_key},
           processor=glassnode_processor, asset_name='BTC', df_col_name='BTC Realized Cap'),
    Metric(base_api_url, 'market', api_key, metric_name='marketcap_realized_usd',
           params={'a': 'ETH', 'api_key': api_key},
           processor=glassnode_processor, asset_name='ETH', df_col_name='ETH Realized Cap'),

    Metric(base_api_url, 'supply', api_key, metric_name='current',
           params={'a': 'TUSD', 'api_key': api_key},
           processor=glassnode_processor, asset_name='TUSD', df_col_name='TUSD Circulating Supply'),
    Metric(base_api_url, 'supply', api_key, metric_name='current',
           params={'a': 'USDT', 'api_key': api_key},
           processor=glassnode_processor, asset_name='USDT', df_col_name='USDT Circulating Supply'),
    Metric(base_api_url, 'supply', api_key, metric_name='current',
           params={'a': 'USDC', 'api_key': api_key},
           processor=glassnode_processor, asset_name='USDC', df_col_name='USDC Circulating Supply'),
    Metric(base_api_url, 'supply', api_key, metric_name='current',
           params={'a': 'BUSD', 'api_key': api_key},
           processor=glassnode_processor, asset_name='BUSD', df_col_name='BUSD Circulating Supply'),

    # # CoinGecko Metrics
    Metric(base_api_cg, 'bitcoin', api_key_cg, metric_name='market_chart',
           params={'vs_currency': 'USD', 'days': 'max', 'x_cg_pro_api_key': api_key_cg},
           processor=cg_processor, asset_name='BTC', df_col_name='BTC Price and Market Cap'),
    Metric(base_api_cg, 'ethereum', api_key_cg, metric_name='market_chart',
           params={'vs_currency': 'USD', 'days': 'max', 'x_cg_pro_api_key': api_key_cg}, processor=cg_processor,
           asset_name='ETH', df_col_name='ETH Price and Market Cap'),

    # Yahoo Finance Metrics
    Metric(base_api_yf, 'chart', api_key_cg, metric_name='^GSPC',
           params={'interval': '1d', 'period1': '473436814', 'period2': f'{unix_today}'}, processor=yf_processor,
           asset_name='S&P500', df_col_name='S&P500 Price'),
    Metric(base_api_yf, 'chart', api_key_cg, metric_name='GC=F',
           params={'interval': '1d', 'period1': '473436814', 'period2': f'{unix_today}'}, processor=yf_processor,
           asset_name='Gold', df_col_name='Gold Price'),

    # FRED Metrics
    Metric(base_api_fred, 'observations', api_key_fred, params={'series_id': 'WALCL',
                                                                'api_key': api_key_fred, 'file_type': 'json'},
           processor=fred_processor, df_col_name='FedBal'),
    Metric(base_api_fred, 'observations', api_key_fred, params={'series_id': 'RRPONTSYD',
                                                                'api_key': api_key_fred, 'file_type': 'json'},
           processor=fred_processor, df_col_name='RRP'),
    Metric(base_api_fred, 'observations', api_key_fred, params={'series_id': 'WTREGEN',
                                                                'api_key': api_key_fred, 'file_type': 'json'},
           processor=fred_processor, df_col_name='TGA'),

    # Coinglass Metrics
    Metric(base_url=base_api_coinglass, endpoint='open_interest_history', api_key=api_key_coinglass,
           processor=coinglass_processor, df_col_name='BTC Open Interest',
           params={'symbol': 'BTC', 'time_type': 'all', 'currency': 'USD'},
           headers={"accept": "application/json", "coinglassSecret": api_key_coinglass}),
    Metric(base_url=base_api_coinglass, endpoint='open_interest_history', api_key=api_key_coinglass,
           processor=coinglass_processor, df_col_name='ETH Open Interest',
           params={'symbol': 'ETH', 'time_type': 'all', 'currency': 'USD'},
           headers={"accept": "application/json", "coinglassSecret": api_key_coinglass}),
    Metric(base_url=base_api_coinglass, endpoint='funding_usd_history', api_key=api_key_coinglass,
           processor=coinglass_processor, df_col_name='BTC Funding Rate',
           params={'symbol': 'BTC', 'time_type': 'h8'},
           headers={"accept": "application/json", "coinglassSecret": api_key_coinglass}),
    Metric(base_url=base_api_coinglass, endpoint='funding_usd_history', api_key=api_key_coinglass,
           processor=coinglass_processor, df_col_name='ETH Funding Rate',
           params={'symbol': 'ETH', 'time_type': 'h8'},
           headers={"accept": "application/json", "coinglassSecret": api_key_coinglass})
]

# Fetch data from APIs
# Todo: figure out why the data is not being cached
with st.spinner('Fetching data from APIs...'):
    data_dict = asyncio.run(prepare_data(metrics))

start_date, end_date, _, _, _, _, _ = st.columns(7)

with start_date:
    zoom_in_date_start = st.date_input('Start Date', datetime.today() - timedelta(days=365))

with end_date:
    zoom_in_date_end = st.date_input('End Date', datetime.today())

# Charts
first_row = st.columns(3)

# Plot settings
distance_from_plot = 0.90
SYNCRACY_COLORS = ['#5218F8', '#F8184E', '#C218F8']

# MVRV
with first_row[0]:
    cols_to_keep = ['BTC MVRV']
    df = merge_dataframes(data_dict, cols_to_keep)
    df = df.dropna()

    fig = px.line(df, x=df.index, y=df.columns, title='BTC: MVRV')
    fig.update_traces(line=dict(color="#5218fa"))
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)

    data_subset = df.loc[zoom_in_date_start:zoom_in_date_end]

    min_val = data_subset['BTC MVRV'].min()
    max_val = data_subset['BTC MVRV'].max()

    fig.update_xaxes(type="date", range=[zoom_in_date_start, zoom_in_date_end])
    fig.update_yaxes(range=[min_val, max_val])

    fig.update_layout(
        title={
            'y': distance_from_plot,
            'x': 0,
            'xanchor': 'left',
            'yanchor': 'top'}
    )

    st.plotly_chart(fig, use_container_width=True)

# BTC Realized Cap HODL Waves
with first_row[1]:
    cols_to_keep = ['BTC Realized Cap HODL Waves']
    df = merge_dataframes(data_dict, cols_to_keep)
    df = df.dropna()

    df_normalized = df.div(df.sum(axis=1), axis=0)

    fig = px.area(df_normalized, labels={"value": "Percent (%)"},
                  title='BTC: Realized Cap HODL Waves')
    fig.update_xaxes(type="date", range=[zoom_in_date_start, zoom_in_date_end])
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    fig.update_yaxes(tickformat=".0%")

    fig.update_layout(
        title={
            'y': distance_from_plot,
            'x': 0,
            'xanchor': 'left',
            'yanchor': 'top'},
        legend_title_text=''
    )

    st.plotly_chart(fig, use_container_width=True)

# BTC Long Term Holder vs Short Term Holder
with first_row[2]:
    cols_to_keep = ['BTC LTH Supply', 'BTC STH Supply']
    df = merge_dataframes(data_dict, cols_to_keep)
    df = df.dropna()

    # Create figure with secondary y-axis
    fig = go.Figure()

    # Add traces
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['BTC LTH Supply'],
            name="BTC LTH Supply",
            line=dict(color='#5218F8'),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['BTC STH Supply'],
            name="BTC STH Supply",
            line=dict(color='#F8184E'),
            yaxis="y2"
        )
    )

    # Create axis objects
    fig.update_layout(
        yaxis=dict(
            title="BTC LTH Supply",

        ),
        yaxis2=dict(
            title="BTC STH Supply",
            anchor="free",
            overlaying="y",
            side="right",
            position=1
        )
    )

    # Update layout properties
    fig.update_layout(
        title_text="BTC: Long-Term vs Short-Term Holder Supply",
        title={
            'y': distance_from_plot,
            'x': 0,
            'xanchor': 'left',
            'yanchor': 'top'},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            itemsizing='constant'
        )
    )
    fig.update_xaxes(type="date", range=[zoom_in_date_start, zoom_in_date_end])
    st.plotly_chart(fig, use_container_width=True)

# Second row
second_row = st.columns(3)

# Net position change
with second_row[0]:
    cols_to_keep = ['BTC Exchange Net Position Change']
    df = merge_dataframes(data_dict, cols_to_keep)
    df = df.dropna()

    data_subset = df.loc[zoom_in_date_start:zoom_in_date_end]

    min_val = data_subset['BTC Exchange Net Position Change'].min()
    max_val = data_subset['BTC Exchange Net Position Change'].max()

    mask_positive = df['BTC Exchange Net Position Change'] >= 0
    mask_negative = df['BTC Exchange Net Position Change'] < 0

    # create figure
    fig = go.Figure()

    # add positive bar
    fig.add_trace(go.Bar(
        x=df.index[mask_positive],
        y=df['BTC Exchange Net Position Change'][mask_positive],
        marker=dict(color='green'),
        name='Positive Net Position'
    ))

    # add negative bar
    fig.add_trace(go.Bar(
        x=df.index[mask_negative],
        y=df['BTC Exchange Net Position Change'][mask_negative],
        marker=dict(color='red'),
        name='Negative Net Position'
    ))

    fig.update_layout(
        title_text="BTC: Exchange Net Position Change",
        title={
            'y': distance_from_plot,
            'x': 0,
            'xanchor': 'left',
            'yanchor': 'top'},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            itemsizing='constant'
        ),
        barmode='relative'
    )

    fig.update_xaxes(type="date", range=[zoom_in_date_start, zoom_in_date_end])
    fig.update_yaxes(range=[min_val, max_val])

    st.plotly_chart(fig, use_container_width=True)

# BTC Correlation with S&P 500 and Gold
with second_row[1]:
    # """
    # Correlations computed using asset returns instead of asset prices
    #
    # Rationale:
    # Returns are unit-less and scale independent, making them easier to compare across different assets.
    # Prices, on the other hand, can vary widely in magnitude across different assets or over time
    # for the same asset, which can distort correlations
    # """
    cols_to_keep = ['BTC Price and Market Cap', 'S&P500 Price', 'Gold Price']
    df = merge_dataframes(data_dict, cols_to_keep)
    df = df[['Price', 'S&P500 Price', 'Gold Price']]
    df = df.replace(0, np.nan)  # replace 0 with NaN
    df = df.dropna()

    data_subset = df.loc[zoom_in_date_start:zoom_in_date_end]

    rolling_window = 90
    returns = df.pct_change().dropna()
    returns['S&P 500'] = returns['Price'].rolling(rolling_window).corr(returns['S&P500 Price'])
    returns['Gold'] = returns['Price'].rolling(rolling_window).corr(returns['Gold Price'])

    min_val = returns[['S&P 500', 'Gold']].min(axis=1)
    max_val = returns[['S&P 500', 'Gold']].max(axis=1)

    fig = px.line(returns[['S&P 500', 'Gold']], title='BTC: Correlation with S&P 500 and Gold',
                  color_discrete_sequence=SYNCRACY_COLORS)
    fig.update_xaxes(type="date", range=[zoom_in_date_start, zoom_in_date_end])
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    fig.update_yaxes(tickformat=".0%")

    fig.update_layout(
        title={
            'y': distance_from_plot,
            'x': 0,
            'xanchor': 'left',
            'yanchor': 'top'},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            itemsizing='constant'
        ),
        legend_title_text=''
    )

    fig.update_xaxes(type="date", range=[zoom_in_date_start, zoom_in_date_end])
    fig.update_yaxes(range=[min_val, max_val])
    st.plotly_chart(fig, use_container_width=True)

# Bitcoin Open Interest to Market Cap Ratio
with second_row[2]:
    cols_to_keep = ['BTC Price and Market Cap', 'BTC Open Interest']
    df = merge_dataframes(data_dict, cols_to_keep)
    df = df.replace(0, np.nan)
    df['OI to Market Cap Ratio'] = df['Aggregated'] / df['Market Cap']
    df.dropna(inplace=True, subset=['OI to Market Cap Ratio'])

    fig = px.line(df, x=df.index, y=df['OI to Market Cap Ratio'],
                  title='BTC: OI To Market Cap Ratio', color_discrete_sequence=SYNCRACY_COLORS)
    fig.update_traces(line=dict(color="#5218fa"))
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)

    data_subset = df.loc[zoom_in_date_start:zoom_in_date_end]

    min_val = data_subset['OI to Market Cap Ratio'].min()
    max_val = data_subset['OI to Market Cap Ratio'].max()

    fig.update_xaxes(type="date", range=[zoom_in_date_start, zoom_in_date_end])
    fig.update_yaxes(range=[min_val, max_val])

    fig.update_layout(
        title={
            'y': distance_from_plot,
            'x': 0,
            'xanchor': 'left',
            'yanchor': 'top'}
    )

    st.plotly_chart(fig, use_container_width=True)

# Third Row
third_row = st.columns(3)

# BTC Total Open Interest
with third_row[0]:
    cols_to_keep = ['BTC Open Interest']
    df = merge_dataframes(data_dict, cols_to_keep)
    cols_to_plot = [col for col in df.columns if col != 'Aggregated']
    df = df.replace(0, np.nan)
    df = df.interpolate(method='linear')

    data_subset = df.loc[zoom_in_date_start:zoom_in_date_end]

    min_val = 0
    max_val = data_subset['Aggregated'].max()

    fig = px.area(df[cols_to_plot], title='BTC: Total OI by Exchange',
                  labels={"value": "Percent (%)"})
    fig.update_layout(xaxis_title=None, yaxis_title=None)

    fig.update_xaxes(type="date", range=[zoom_in_date_start, zoom_in_date_end])
    fig.update_yaxes(range=[min_val, max_val])

    fig.update_layout(
        title={
            'y': distance_from_plot,
            'x': 0,
            'xanchor': 'left',
            'yanchor': 'top'},
        legend_title_text=''
    )

    st.plotly_chart(fig, use_container_width=True)

# BTC Funding Rate
with third_row[1]:
    cols_to_keep = ['BTC Funding Rate']
    df = merge_dataframes(data_dict, cols_to_keep)

    data_subset = df.loc[zoom_in_date_start:zoom_in_date_end]

    min_val = data_subset['Aggregated'].min()
    max_val = data_subset['Aggregated'].max()

    mask_positive = df['Aggregated'] >= 0
    mask_negative = df['Aggregated'] < 0

    # create figure
    fig = go.Figure()

    # add positive bar
    fig.add_trace(go.Bar(
        x=df.index[mask_positive],
        y=df['Aggregated'][mask_positive],
        marker=dict(color='green'),
        name='Positive Funding Rate'
    ))

    # add negative bar
    fig.add_trace(go.Bar(
        x=df.index[mask_negative],
        y=df['Aggregated'][mask_negative],
        marker=dict(color='red'),
        name='Negative Funding Rate'
    ))

    fig.update_layout(
        title_text="BTC: Aggregate Funding Rate (8 hr)",
        title={
            'y': distance_from_plot,
            'x': 0,
            'xanchor': 'left',
            'yanchor': 'top'},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            itemsizing='constant'
        ),
        barmode='relative'
    )

    fig.update_xaxes(type="date", range=[zoom_in_date_start, zoom_in_date_end])
    fig.update_yaxes(range=[min_val, max_val])

    st.plotly_chart(fig, use_container_width=True)

# BTC vs Liquidity Index
with third_row[2]:
    cols_to_keep = ['BTC Price and Market Cap', 'FedBal', 'RRP', 'TGA']
    df = merge_dataframes(data_dict, cols_to_keep)
    df = df.replace(to_replace=0, method='ffill')
    df.dropna(inplace=True)

    cut_off_date = '2014-03-01'
    df = df.loc[cut_off_date:]

    # Scale Liquidity Index
    df['FedBal'] = df['FedBal'] * 1_000_000
    df['RRP'] = df['RRP'] * 1_000_000_000
    df['TGA'] = df['TGA'] * 1_000_000_000

    df['Liquidity Index'] = df['FedBal'] - df['RRP'] - df['TGA']

    data_subset = df.loc[zoom_in_date_start:zoom_in_date_end]

    liq_min_val = data_subset['Liquidity Index'].min()
    liq_max_val = data_subset['Liquidity Index'].max()

    price_min_val = data_subset['Price'].min()
    price_max_val = data_subset['Price'].max()

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Price'],
            mode='lines',
            line=dict(color='#F8184E'),
            name='BTC Price'
        ))

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Liquidity Index'],
            mode='lines',
            name='Liquidty Index',
            yaxis="y2",
            line=dict(color='#5218F8')
        ))

    # Create axis objects
    fig.update_layout(
        yaxis=dict(
            title="BTC Price",

        ),
        yaxis2=dict(
            title="Liquidity Index",
            anchor="free",
            overlaying="y",
            side="right",
            position=1
        )
    )

    # Update layout properties
    fig.update_layout(
        title_text="BTC: Price vs Liquidity Index",
        title={
            'y': distance_from_plot,
            'x': 0,
            'xanchor': 'left',
            'yanchor': 'top'},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            itemsizing='constant'
        )
    )
    fig.update_xaxes(type="date", range=[zoom_in_date_start, zoom_in_date_end])
    fig.update_yaxes(range=[price_min_val, price_max_val])
    fig.update_yaxes(range=[liq_min_val, liq_max_val], secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

# Fourth row
fourth_row = st.columns(3)

# ETH Active Addresses
with fourth_row[0]:
    cols_to_keep = ['ETH Active Addresses Count']
    df = merge_dataframes(data_dict, cols_to_keep)
    df = df.dropna()

    fig = px.line(df, x=df.index, y=df.columns, title='ETH: Active Addresses')
    fig.update_traces(line=dict(color="#5218fa"))
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)

    data_subset = df.loc[zoom_in_date_start:zoom_in_date_end]

    min_val = data_subset['ETH Active Addresses Count'].min()
    max_val = data_subset['ETH Active Addresses Count'].max()

    fig.update_xaxes(type="date", range=[zoom_in_date_start, zoom_in_date_end])
    fig.update_yaxes(range=[min_val, max_val])

    fig.update_layout(
        title={
            'y': distance_from_plot,
            'x': 0,
            'xanchor': 'left',
            'yanchor': 'top'}
    )

    st.plotly_chart(fig, use_container_width=True)

# ETH gas price vs transactions count
with fourth_row[1]:
    cols_to_keep = ['ETH Gas Price Mean', 'ETH Transactions Count']
    df = merge_dataframes(data_dict, cols_to_keep)
    df = df.dropna()

    data_subset = df.loc[zoom_in_date_start:zoom_in_date_end]

    gas_min_val = data_subset['ETH Gas Price Mean'].min()
    gas_max_val = data_subset['ETH Gas Price Mean'].max()

    tx_min_val = data_subset['ETH Transactions Count'].min()
    tx_max_val = data_subset['ETH Transactions Count'].max()

    # create figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['ETH Gas Price Mean'],
            fill='tozeroy',
            mode='lines',
            line=dict(color='#F8184E'),
            name='ETH Gas Price'
        ))

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['ETH Transactions Count'],
            mode='lines',
            name='ETH Transactions Count',
            yaxis="y2",
            line=dict(color='#5218F8')
        ))

    # Create axis objects
    fig.update_layout(
        yaxis=dict(
            title="ETH Gas Price",

        ),
        yaxis2=dict(
            title="ETH Transactions Count",
            anchor="free",
            overlaying="y",
            side="right",
            position=1
        )
    )

    # Update layout properties
    fig.update_layout(
        title_text="ETH: Gas Prices and Transaction Count",
        title={
            'y': distance_from_plot,
            'x': 0,
            'xanchor': 'left',
            'yanchor': 'top'},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            itemsizing='constant'
        )
    )
    fig.update_xaxes(type="date", range=[zoom_in_date_start, zoom_in_date_end])
    fig.update_yaxes(range=[gas_min_val, gas_max_val])
    fig.update_yaxes(range=[tx_min_val, tx_max_val], secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

# ETH Transactions by Type
with fourth_row[2]:
    cols_to_keep = ['ETH Transaction Types Breakdown']
    df = merge_dataframes(data_dict, cols_to_keep)
    df = df.dropna()

    df_normalized = df.div(df.sum(axis=1), axis=0)

    fig = px.area(df_normalized, labels={"value": "Percent (%)"},
                  title='ETH: Gas Used by Transaction Type')
    fig.update_xaxes(type="date", range=[zoom_in_date_start, zoom_in_date_end])
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    fig.update_yaxes(tickformat=".0%")

    fig.update_layout(
        title={
            'y': distance_from_plot,
            'x': 0,
            'xanchor': 'left',
            'yanchor': 'top'},
        legend_title_text=''
    )

    st.plotly_chart(fig, use_container_width=True)

# Fifth row
fifth_row = st.columns(3)

# ETH Correlations to S&P500 and Gold
with fifth_row[0]:
    # """
    # Correlations computed using asset returns instead of asset prices
    #
    # Rationale:
    # Returns are unit-less and scale independent, making them easier to compare across different assets.
    # Prices, on the other hand, can vary widely in magnitude across different assets or over time
    # for the same asset, which can distort correlations
    # """
    cols_to_keep = ['ETH Price and Market Cap', 'S&P500 Price', 'Gold Price']
    df = merge_dataframes(data_dict, cols_to_keep)
    df = df[['Price', 'S&P500 Price', 'Gold Price']]
    df = df.replace(0, np.nan)  # replace 0 with NaN
    df = df.dropna()

    data_subset = df.loc[zoom_in_date_start:zoom_in_date_end]

    rolling_window = 90
    returns = df.pct_change().dropna()
    returns['S&P 500'] = returns['Price'].rolling(rolling_window).corr(returns['S&P500 Price'])
    returns['Gold'] = returns['Price'].rolling(rolling_window).corr(returns['Gold Price'])

    min_val = returns[['S&P 500', 'Gold']].min(axis=1)
    max_val = returns[['S&P 500', 'Gold']].max(axis=1)

    fig = px.line(returns[['S&P 500', 'Gold']], title='ETH: Correlation with S&P 500 and Gold',
                  color_discrete_sequence=SYNCRACY_COLORS)
    fig.update_xaxes(type="date", range=[zoom_in_date_start, zoom_in_date_end])
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    fig.update_yaxes(tickformat=".0%")

    fig.update_layout(
        title={
            'y': distance_from_plot,
            'x': 0,
            'xanchor': 'left',
            'yanchor': 'top'},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            itemsizing='constant'
        ),
        legend_title_text=''
    )

    fig.update_xaxes(type="date", range=[zoom_in_date_start, zoom_in_date_end])
    fig.update_yaxes(range=[min_val, max_val])
    st.plotly_chart(fig, use_container_width=True)

# ETH Total Open Interest
with fifth_row[1]:
    cols_to_keep = ['ETH Open Interest']
    df = merge_dataframes(data_dict, cols_to_keep)
    cols_to_plot = [col for col in df.columns if col != 'Aggregated']
    df = df.replace(0, np.nan)
    df = df.interpolate(method='linear')

    data_subset = df.loc[zoom_in_date_start:zoom_in_date_end]

    min_val = 0
    max_val = data_subset['Aggregated'].max()

    fig = px.area(df[cols_to_plot], title='ETH: Total OI by Exchange',
                  labels={"value": "Percent (%)"})
    fig.update_layout(xaxis_title=None, yaxis_title=None)

    fig.update_xaxes(type="date", range=[zoom_in_date_start, zoom_in_date_end])
    fig.update_yaxes(range=[min_val, max_val])

    fig.update_layout(
        title={
            'y': distance_from_plot,
            'x': 0,
            'xanchor': 'left',
            'yanchor': 'top'},
        legend_title_text=''
    )

    st.plotly_chart(fig, use_container_width=True)

# ETH Funding Rate
with fifth_row[2]:
    cols_to_keep = ['ETH Funding Rate']
    df = merge_dataframes(data_dict, cols_to_keep)

    data_subset = df.loc[zoom_in_date_start:zoom_in_date_end]

    min_val = data_subset['Aggregated'].min()
    max_val = data_subset['Aggregated'].max()

    mask_positive = df['Aggregated'] >= 0
    mask_negative = df['Aggregated'] < 0

    # create figure
    fig = go.Figure()

    # add positive bar
    fig.add_trace(go.Bar(
        x=df.index[mask_positive],
        y=df['Aggregated'][mask_positive],
        marker=dict(color='green'),
        name='Positive Funding Rate'
    ))

    # add negative bar
    fig.add_trace(go.Bar(
        x=df.index[mask_negative],
        y=df['Aggregated'][mask_negative],
        marker=dict(color='red'),
        name='Negative Funding Rate'
    ))

    fig.update_layout(
        title_text="ETH: Aggregate Funding Rate (8 hr)",
        title={
            'y': distance_from_plot,
            'x': 0,
            'xanchor': 'left',
            'yanchor': 'top'},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            itemsizing='constant'
        ),
        barmode='relative'
    )

    fig.update_xaxes(type="date", range=[zoom_in_date_start, zoom_in_date_end])
    fig.update_yaxes(range=[min_val, max_val])

    st.plotly_chart(fig, use_container_width=True)

# Sixth row
sixth_row = st.columns(3)

# S&P 500 vs Liquidity
with sixth_row[0]:
    cols_to_keep = ['S&P500 Price', 'FedBal', 'RRP', 'TGA']
    df = merge_dataframes(data_dict, cols_to_keep)
    df = df.replace(to_replace=0, method='ffill')
    df.dropna(inplace=True)

    cut_off_date = '2014-03-01'
    df = df.loc[cut_off_date:]

    # Scale Liquidity Index
    df['FedBal'] = df['FedBal'] * 1_000_000
    df['RRP'] = df['RRP'] * 1_000_000_000
    df['TGA'] = df['TGA'] * 1_000_000_000

    df['Liquidity Index'] = df['FedBal'] - df['RRP'] - df['TGA']

    data_subset = df.loc[zoom_in_date_start:zoom_in_date_end]

    liq_min_val = data_subset['Liquidity Index'].min()
    liq_max_val = data_subset['Liquidity Index'].max()

    price_min_val = data_subset['S&P500 Price'].min()
    price_max_val = data_subset['S&P500 Price'].max()

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['S&P500 Price'],
            mode='lines',
            line=dict(color='#F8184E'),
            name='S&P 500 Price'
        ))

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Liquidity Index'],
            mode='lines',
            name='Liquidty Index',
            yaxis="y2",
            line=dict(color='#5218F8')
        ))

    # Create axis objects
    fig.update_layout(
        yaxis=dict(
            title="S&P 500 Price",

        ),
        yaxis2=dict(
            title="Liquidity Index",
            anchor="free",
            overlaying="y",
            side="right",
            position=1
        )
    )

    # Update layout properties
    fig.update_layout(
        title_text="S&P 500 Price vs Liquidity Index",
        title={
            'y': distance_from_plot,
            'x': 0,
            'xanchor': 'left',
            'yanchor': 'top'},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            itemsizing='constant'
        )
    )
    fig.update_xaxes(type="date", range=[zoom_in_date_start, zoom_in_date_end])
    fig.update_yaxes(range=[price_min_val, price_max_val])
    fig.update_yaxes(range=[liq_min_val, liq_max_val], secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

# Stablecoin Transfer Volume
with sixth_row[1]:
    cols_to_keep = ['USDT Transfers Volume', 'USDC Transfers Volume', 'BUSD Transfers Volume',
                    'DAI Transfers Volume']
    df = merge_dataframes(data_dict, cols_to_keep)
    df.columns = ['USDT', 'USDC', 'BUSD', 'DAI']
    data_subset = df.loc[zoom_in_date_start:zoom_in_date_end]

    min_val = 0
    max_val = data_subset.max()

    fig = px.line(df, title='Stablecoin Transfer Volume')
    fig.update_layout(xaxis_title=None, yaxis_title=None)

    fig.update_xaxes(type="date", range=[zoom_in_date_start, zoom_in_date_end])
    fig.update_yaxes(range=[min_val, max_val])

    fig.update_layout(
        title={
            'y': distance_from_plot,
            'x': 0,
            'xanchor': 'left',
            'yanchor': 'top'},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            itemsizing='constant'
        ),
        legend_title_text=''
    )

    st.plotly_chart(fig, use_container_width=True)

# Market Realized Value Net Position Change
with sixth_row[2]:
    # BTC+ETH Net position change = diff(BTC+ETH Realized Cap, 30)
    # Aggregate major asset realized cap = BTC+ETH+TUSD+USDT+USDC+BUSD
    # Aggregate value net position change = diff(Aggregate major asset realized cap, 30)
    cols_to_keep = ['BTC Realized Cap', 'ETH Realized Cap', 'TUSD Circulating Supply',
                    'USDT Circulating Supply', 'USDC Circulating Supply', 'BUSD Circulating Supply']
    df = merge_dataframes(data_dict, cols_to_keep)

    df['BTC+ETH Realized Cap'] = df['BTC Realized Cap'] + df['ETH Realized Cap']
    df['BTC+ETH Net Position Change'] = df['BTC+ETH Realized Cap'].diff(30)

    df['Stablecoin Net Position Change'] = (df['TUSD Circulating Supply'] + df['USDT Circulating Supply'] + df[
        'USDC Circulating Supply'] + df['BUSD Circulating Supply']).diff(30)

    df['Aggregate Major Asset Realized Cap'] = df['BTC+ETH Realized Cap'] + df['TUSD Circulating Supply'] + df[
        'USDT Circulating Supply'] + df['USDC Circulating Supply'] + df['BUSD Circulating Supply']
    df['Aggregate Major Asset Net Position Change'] = df['Aggregate Major Asset Realized Cap'].diff(30)

    cut_off_date = '2020-01-01'
    df = df.loc[cut_off_date:]

    mask_positive = df['Aggregate Major Asset Net Position Change'] >= 0
    mask_negative = df['Aggregate Major Asset Net Position Change'] < 0

    # create figure
    fig = go.Figure()

    # add positive bar
    fig.add_trace(go.Bar(
        x=df.index[mask_positive],
        y=df['Aggregate Major Asset Net Position Change'][mask_positive],
        marker=dict(color='green'),
        name='Positive 30-day Inflow'
    ))

    # add negative bar
    fig.add_trace(go.Bar(
        x=df.index[mask_negative],
        y=df['Aggregate Major Asset Net Position Change'][mask_negative],
        marker=dict(color='red'),
        name='Negative 30-day Inflow'
    ))

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['BTC+ETH Net Position Change'],
            mode='lines',
            name='BTC+ETH Net Position Change',
            line=dict(color='#5218F8')
        ))

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Stablecoin Net Position Change'],
            mode='lines',
            name='Stablecoin Net Position Change',
            line=dict(color='#F8184E')
        ))

    fig.update_layout(
        title_text="Aggregate Market Realized Value Net Position Change",
        title={
            'y': distance_from_plot,
            'x': 0,
            'xanchor': 'left',
            'yanchor': 'top'},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=.99,
            xanchor="center",
            x=0.5,
            itemsizing='constant'
        ),
        barmode='relative'
    )

    fig.update_xaxes(type="date", range=[zoom_in_date_start, zoom_in_date_end])
    st.plotly_chart(fig, use_container_width=True)
