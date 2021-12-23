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
    d = {
        'start':    [],     # start time of product, datetime() object
        'end':      [],     # end time of product, datetime() object
        'price':    []      # price of product
    }

    return pd.DataFrame(data=d)


if __name__ == "__main__":
    md = MarketData()
