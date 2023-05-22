import asyncio

from aiocache import Cache
from aiocache import cached
from aiocache.serializers import PickleSerializer


def big_number_formatter(x):
    """The two args are the value and tick position."""
    formatter_thresholds = 1_000_000_000
    if x < formatter_thresholds:
        return '${:1.2f}M'.format(x * 1e-6)
    else:
        return '${:1.2f}B'.format(x * 1e-9)


@cached(ttl=21600, serializer=PickleSerializer(), cache=Cache.MEMORY)  # 6 hours
async def prepare_data(metrics):  # noqa
    """
    Fetches data from the API for the given metrics
    :param metrics: list of Metric objects
    :return: pandas DataFrame
    """
    tasks = [metric.fetch_data() for metric in metrics]
    results = await asyncio.gather(*tasks)
    data = {key: value for result in results for key, value in result.items()}
    return data


def merge_dataframes(data_dict, metric_names):
    """
    Merges multiple DataFrames into a single DataFrame
    :param data_dict: dict with metric names as keys and DataFrames as values
    :param metric_names: list of metric names to merge
    :return: merged DataFrame
    """
    # Start with the first metric's DataFrame
    try:
        merged_df = data_dict[metric_names[0]]
    except KeyError:
        raise ValueError(f'Metric not found in data dictionary: {metric_names[0]}')

    # Merge the rest of the DataFrames
    for metric_name in metric_names[1:]:
        try:
            df = data_dict[metric_name]
        except KeyError:
            raise ValueError(f'Metric not found in data dictionary: {metric_name}')

        # Check for duplicate column names before merging
        common_cols = set(merged_df.columns) & set(df.columns)
        for col in common_cols:
            df = df.rename(columns={col: f'{metric_name}_{col}'})

        # Merge the DataFrame, handling missing values
        merged_df = merged_df.join(df, how='outer')
        # merged_df = pd.merge(merged_df, df, how='outer', left_index=True, right_index=True)

    # Handle missing values in the merged DataFrame
    merged_df = merged_df.fillna(0)  # replace NaNs with 0; change this as needed

    return merged_df
