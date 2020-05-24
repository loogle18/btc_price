import os
import json
import glob
import pandas as pd
from pytrends import dailydata
from functools import partial
from multiprocessing import Pool, cpu_count


def get_daily(keyword, group, ys, ms, ye, me):
    target_column = "%s_unscaled" % keyword
    df = dailydata.get_daily_data(keyword, int(ys), int(ms), int(ye), int(me), geo="")
    df.drop(df.columns.difference(["date", target_column]), 1, inplace=True)
    df.rename(columns={target_column: keyword}, inplace=True)
    df.to_csv("data/original/trends/%s/%s-%s-%s-%s-%s.csv" % (group, keyword, ys, ms, ye, me))


def get_daily_trends(group, ys, ms, ye, me):
    global trends
    pool = Pool(processes=cpu_count())
    pool.map(partial(get_daily, group=group, ys=ys, ms=ms, ye=ye, me=me), trends[group])
    pool.close()
    pool.join()


def get_merged_df(files, on="date"):
    final_df = pd.read_csv(files[0])
    for file in files[1:]:
        df = pd.read_csv(file)
        final_df = pd.merge(final_df, df, on=on)
    return final_df


def combine_data(group, ys, ms, ye, me):
    folder = "data/original/trends/%s" % group
    files_path = "%s/*.csv" % folder
    files = [fp for fp in glob.glob(files_path) if not fp.startswith("%s/combined" % folder)]
    df = get_merged_df(files)
    os.system("rm -rf %s" % files_path)
    df.to_csv("data/original/trends/%s/combined-%s-%s-%s-%s.csv" % (group, ys, ms, ye, me), index=None)


def sum_data(group, ys, ms, ye, me):
    files_path = "data/original/trends/%s/combined-%s-%s-%s-%s.csv" % (group, ys, ms, ye, me)
    df = pd.read_csv(files_path)
    df[group] = df.sum(axis=1, skipna=True, numeric_only=True)
    df.drop(df.columns.difference(["date", group]), 1, inplace=True)
    df.to_csv("data/original/trends/%s/combined-%s-%s-%s-%s-sum.csv" % (group, ys, ms, ye, me), index=None)


def process_group(group, ys, ms, ye, me):
    # get_daily_trends(group, ys, ms, ye, me)
    combine_data(group, ys, ms, ye, me)
    sum_data(group, ys, ms, ye, me)


def combine_trend_sums(ys, ms, ye, me):
    files_path = "data/original/trends/**/combined-*-sum.csv"
    files = glob.glob(files_path)
    df = get_merged_df(files)
    os.system("rm -rf %s" % files_path)
    df.to_csv("data/clean/trends-%s-%s-%s-%s.csv" % (ys, ms, ye, me), index=None)


def main():
    global trends
    ys, ms, ye, me = "2018", "02", "2020", "04"
    trends = json.load(open("trends.json", "r"))
    process_group("negative", ys, ms, ye, me)
    process_group("neutral", ys, ms, ye, me)
    process_group("positive", ys, ms, ye, me)
    combine_trend_sums(ys, ms, ye, me)


if __name__ == "__main__":
    main()
    # ys, ms, ye, me = "2018", "02", "2020", "04"
    # trends = json.load(open("trends.json", "r"))
    # for group in ["negative", "neutral", "positive"]:
    #     folder = "data/original/trends/%s" % group
    #     for keyword in trends[group]:
    #         target_column = "%s_unscaled" % keyword
    #         df = pd.read_csv("%s/%s-%s-%s-%s-%s-all.csv" % (folder, keyword, ys, ms, ye, me))
    #         # df.to_csv("data/original/trends/%s/%s-%s-%s-%s-%s-all.csv" % (group, keyword, ys, ms, ye, me))
    #         df.drop(df.columns.difference(["date", target_column]), 1, inplace=True)
    #         df.rename(columns={target_column: keyword}, inplace=True)
    #         df.to_csv("data/original/trends/%s/%s-%s-%s-%s-%s.csv" % (group, keyword, ys, ms, ye, me), index=None)
    