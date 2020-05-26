import trends
import blockchain
import coinmarketcap
import fear_and_greed
import pandas as pd
from datetime import date


COLUMNS = [
    "date", "open", "high", "low", "close",
    "market_cap_btc", "market_cap_all", "volume_btc",
    "volume_all", "btc_cap_pct", "btc_volume_pct",
    "fng", "thr", "difficulty", "transactions",
    "neutral", "positive", "negative"
]


def get_data(start_date, end_date):
    df_trends = trends.main(start_date, end_date)
    df_trends.to_csv("data/clean/trends.csv", index=None)
    df_blockchain = blockchain.main()
    df_blockchain.to_csv("data/clean/blockchain.csv", index=None)
    df_coinmarketcap = coinmarketcap.main(start_date, end_date)
    df_coinmarketcap.to_csv("data/clean/coinmarketcap.csv", index=None)
    df_fear_and_greed = fear_and_greed.main()
    df_fear_and_greed.to_csv("data/clean/fear_and_greed.csv", index=None)
    df1 = pd.merge(df_trends, df_blockchain, on="date")
    df2 = pd.merge(df_coinmarketcap, df_fear_and_greed, on="date")
    df = pd.merge(df1, df2, on="date")
    df.dropna(inplace=True)
    return df


def main(start_date, end_date):
    df_current = pd.read_csv("data/clean/final-new.csv", names=COLUMNS, skiprows=1, low_memory=False)
    df_new = get_data(start_date, end_date)
    df_new = df_new[COLUMNS]
    df = pd.concat([df_current, df_new], ignore_index=True).drop_duplicates(["date"], keep="last")
    df.to_csv("data/clean/final-new.csv", index=None)


if __name__ == "__main__":
    main("2020-05-18", date.today().strftime("%Y-%m-%d"))
