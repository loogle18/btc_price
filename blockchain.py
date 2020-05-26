import requests
import pandas as pd
import decorators


def request_chart_data(endpoint):
    params = {
        # "daysAverageString": "7D",
        "timespan": "3years",
        "sampled": "false",
        "metadata": "false",
        "cors": "true",
        "format": "json"
    }
    res = requests.get(url="https://api.blockchain.info/charts/%s" % endpoint,
                       params=params)
    return res.json()


def get_thr_data():
    data = request_chart_data("hash-rate")
    df = pd.DataFrame(data["values"])
    df.rename(columns={"x": "date", "y": "thr"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], utc=True, unit="s").dt.strftime('%Y-%m-%d')
    return df


def get_difficulty_data():
    data = request_chart_data("difficulty")
    df = pd.DataFrame(data["values"])
    df.rename(columns={"x": "date", "y": "difficulty"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], utc=True, unit="s").dt.strftime('%Y-%m-%d')
    return df


def get_transactions_data():
    data = request_chart_data("n-transactions")
    df = pd.DataFrame(data["values"])
    df.rename(columns={"x": "date", "y": "transactions"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], utc=True, unit="s").dt.strftime('%Y-%m-%d')
    return df


@decorators.error_handler
def main(save=False):
    df1 = get_thr_data()
    df2 = get_difficulty_data()
    df3 = get_transactions_data()
    df4 = pd.merge(df1, df2, on="date")
    df = pd.merge(df4, df3, on="date")
    df[["thr", "difficulty", "transactions"]] = df[["thr", "difficulty", "transactions"]].applymap(int)
    if save:
        df.to_csv("data/clean/blockchain.csv", index=None)
    return df
