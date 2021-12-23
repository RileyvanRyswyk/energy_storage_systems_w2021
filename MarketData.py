import datetime
import re

import numpy as np
import pandas as pd
import glob


class MarketData:

    DATA_PATH = 'data/market'

    def __init__(self, start=None, end=None):
        # load all FCR files in the data path
        frames = []
        files = glob.glob("{}/*FCR*.xlsx".format(self.DATA_PATH))
        for file in files:
            frames.append(load_data(file))
        df = pd.concat(frames)

        self.df = map_market_data(df)


def load_data(file_path):
    return pd.read_excel(
        file_path,
        sheet_name=0,
        header=0,
        usecols="A,E,Q",
        names=["date", "product", "price"],
        parse_dates=[1],
    )


def map_market_data(df):
    start = np.zeros(len(df), dtype=datetime.time)
    end = np.zeros(len(df), dtype=datetime.time)

    for index, product_name in enumerate(df["product"]):
        p = re.compile(r'NEGPOS_(\d+)_(\d+)')
        start_hour, end_hour = p.match(product_name).groups()

        start[index] = df["date"].iloc[index] + datetime.timedelta(hours=int(start_hour))
        end[index] = df["date"].iloc[index] + datetime.timedelta(hours=int(end_hour))

    d = {
        'start':    start,                  # start time of product, datetime() object
        'end':      end,                    # end time of product, datetime() object
        'price':    df["price"].values      # price of product
    }

    return pd.DataFrame(data=d)


if __name__ == "__main__":
    md = MarketData()
