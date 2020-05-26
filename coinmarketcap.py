import requests
import pandas as pd
from datetime import datetime, timedelta
import decorators


def request_global_data(start_time, end_time):
    params = {
        "format": "chart",
        "interval": "1d",
        "time_start": start_time,
        "time_end": end_time
    }
    res = requests.get(url="https://web-api.coinmarketcap.com/v1.1/global-metrics/quotes/historical",
                       params=params)
    return res.json()


def request_btc_data(start_time, end_time):
    params = {
        "convert": "USD",
        "slug": "bitcoin",
        "time_start": start_time,
        "time_end": end_time
    }
    res = requests.get(url="https://web-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical",
                       params=params)
    return res.json()


def dt_to_ts(dt_str):
    return int((pd.to_datetime(dt_str) - datetime(1970, 1, 1)).total_seconds())


def parse_global_data(data):
    result = []
    for timestamp, values in data.items():
        result.append({
            "date": pd.to_datetime(timestamp, utc=True).strftime('%Y-%m-%d'),
            "market_cap_all": values[0],
            "volume_all": values[1]
        })
    return result


@decorators.error_handler
def parse_btc_data(data):
    result = []
    for value in data:
        value = value["quote"]["USD"]
        result.append({
            "date": pd.to_datetime(value["timestamp"], utc=True).strftime('%Y-%m-%d'),
            "open": value["open"],
            "high": value["high"],
            "low": value["low"],
            "close": value["close"],
            "volume_btc": value["volume"],
            "market_cap_btc": value["market_cap"]
        })
    return result


def get_global_data(start_time, end_time):
    json = request_global_data(start_time, end_time)
    data = parse_global_data(json["data"])
    df = pd.DataFrame(data, columns=["date", "market_cap_all", "volume_all"])
    return df


def get_btc_data(start_time, end_time):
    json = request_btc_data(start_time, end_time)
    data = parse_btc_data(json["data"]["quotes"])
    df = pd.DataFrame(data, columns=["date", "open", "high", "low", "close", "volume_btc", "market_cap_btc"])
    return df


@decorators.error_handler
def main(start_date, end_date, save=False):
    # start_date = str((datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=1)).date())
    start_time = dt_to_ts(start_date)
    end_time = dt_to_ts(end_date)
    df1 = get_btc_data(start_time, end_time)
    df2 = get_global_data(start_time, end_time)
    df = pd.merge(df1, df2, on="date")
    df["btc_cap_pct"] = (df["market_cap_btc"] * 100 / df["market_cap_all"]).astype(float)
    df["btc_volume_pct"] = (df["volume_btc"] * 100 / df["volume_all"]).astype(float)
    # df.drop(["market_cap_btc", "market_cap_all", "volume_btc", "volume_all"], axis=1, inplace=True)
    df[["market_cap_btc", "market_cap_all", "volume_btc", "volume_all"]] = df[["market_cap_btc", "market_cap_all", "volume_btc", "volume_all"]].applymap(int)
    if save:
        df.to_csv("data/clean/coinmarketcap.csv", index=None)
    return df
