import datetime
import http.client
import io
import json
import math
import re

import numpy as np
import pandas as pd
import glob

DATA_PATH = 'data/market'


class MarketData:

    def __init__(self, start=None, end=None):
        self.da_df = load_day_ahead_data()
        self.fcr_df = map_market_data(load_fcr_data())


# load all day ahead files in the data path
def load_day_ahead_data():
    frames = []
    files = glob.glob("{}/*day_ahead*.csv".format(DATA_PATH))
    for file in files:
        frames.append(pd.read_csv(
            file,
            header=0,
            index_col="start",
            parse_dates=[0],
        ))
    return pd.concat(frames)


# load all FCR files in the data path
def load_fcr_data():
    frames = []
    files = glob.glob("{}/*FCR*.xlsx".format(DATA_PATH))
    for file in files:
        frames.append(pd.read_excel(
            file,
            sheet_name=0,
            header=0,
            usecols="A,E,Q",
            names=["date", "product", "price"],
            parse_dates=[1],
        ))
    df = pd.concat(frames)

    return df


def map_market_data(df):
    start = np.zeros(len(df), dtype=datetime.time)
    end = np.zeros(len(df), dtype=datetime.time)

    for index, product_name in enumerate(df["product"]):
        p = re.compile(r'NEGPOS_(\d+)_(\d+)')
        start_hour, end_hour = p.match(product_name).groups()

        start[index] = df["date"].iloc[index] + datetime.timedelta(hours=int(start_hour))
        end[index] = df["date"].iloc[index] + datetime.timedelta(hours=int(end_hour))

    d = {
        # 'start':    start,                  # start time of product, datetime() object
        # 'end':      end,                    # end time of product, datetime() object
        'price':    df["price"].values      # price of product
    }

    return pd.DataFrame(data=d, index=start)


def compile_day_ahead_prices(start, end):
    max_batch_duration = datetime.timedelta(days=14)
    n_batch = math.ceil((end-start) / max_batch_duration)

    df = None
    for n in range(0, n_batch):
        start_n = start + n * max_batch_duration
        end_n = min(end, start_n + max_batch_duration)
        response = request_smard_data(
            format_timestamp_ms(start_n),
            format_timestamp_ms(end_n)
        )

        df_n = pd.read_csv(
            io.StringIO(response),
            names=["date", "time", "price"],
            parse_dates={"start": ["date", "time"]},
            sep=';',
            header=0
        )

        if df is None:
            df = df_n
        else:
            df = df.append(df_n)

    df.to_csv('{}/day_ahead_from_{}_to_{}.csv'.format(DATA_PATH, start.date(), end.date()), index=False)


def format_timestamp_ms(ts):
    return round(ts.timestamp() * 1000)


def request_smard_data(from_timestamp, to_timestamp):
    request_body = {
        "request_form": [
            {
                "format": "CSV",
                "timestamp_from": from_timestamp,
                "timestamp_to": to_timestamp,
                "language": "en",
                "moduleIds": [
                    8004169
                ],
                "type": "discrete",
                "region": "DE"
            }
        ]
    }

    headers = {"Content-type": "application/json",
               "Accept": "application/json, text/plain, */*"}
    conn = http.client.HTTPSConnection("www.smard.de", port=443)
    conn.request("POST", "/nip-download-manager/nip/download/market-data", json.dumps(request_body), headers)
    response = conn.getresponse()
    print(response.status, response.reason)
    data = response.read().decode('utf-8')
    conn.close()

    return data


if __name__ == "__main__":
    # md = MarketData()
    compile_day_ahead_prices(
        start=datetime.datetime(year=2021, month=11, day=1),
        end=datetime.datetime(year=2021, month=12, day=1)
    )
