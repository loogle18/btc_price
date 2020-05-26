import requests
import pandas as pd
import decorators


def request_fng_data():
    params = {
        "limit": "0",
        "date_format": "world",
        "format": "json"
    }
    res = requests.get(url="https://api.alternative.me/fng/",
                       params=params)
    return res.json()


def parse_fng_data(data):
    result = []
    for value in reversed(data):
        result.append({
            "date": pd.to_datetime(value["timestamp"]).strftime('%Y-%m-%d'),
            "fng": int(value["value"])
        })
    return result


def get_fng_data():
    data = request_fng_data()
    return pd.DataFrame(parse_fng_data(data["data"]))


@decorators.error_handler
def main(save=False):
    df = get_fng_data()
    if save:
        df.to_csv("data/clean/fng.csv", index=None)
    return df
