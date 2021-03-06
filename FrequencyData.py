import glob
from datetime import timedelta
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class FrequencyData:

    # Data set from regelleistung.net for PQ pre-qualification
    # January 2019
    # https://www.regelleistung.net/ext/download/example_of_frequency
    PQ_DATA = "data/201901_Frequenz.csv"

    # Data set from Denmark Technical University (DTU)
    # Grid Frequency Measurements of the Continental European Power System during 2019
    # https://data.dtu.dk/articles/dataset/Grid_Frequency_Measurements_of_the_Continental_European_Power_System_during_2019/12758429
    DTU_DATA = "data/freq_DK1_2019.csv"

    DATA_PATH = 'data'

    # # Nominal frequency [Hz]
    F_NOMINAL = 50

    def __init__(self, data_src=None):
        self.df = None          # data frame
        self.time_step = 0      # data time step in seconds

        if data_src == self.DTU_DATA:
            self.load_dtu_data()
        elif data_src is not None:
            self.load_freq_data_files(data_src)
        else:
            raise Exception('Data source not provided')

        # self.compute_delta_f()

        # diagnostics
        self.df.info(verbose=False, memory_usage="deep")
        print(f'Frequencies range between {self.df["freq"].min():.3f} and {self.df["freq"].max():.3f}')

    def load_freq_data_files(self, data_file_mask):
        self.time_step = 1  # data at 1 second intervals
        frames = []
        files = glob.glob("{}".format(data_file_mask))

        for file in files:
            frames.append(pd.read_csv(
                file,
                names=["date", "time", "freq"],
                usecols=[0, 1, 3],
                parse_dates={"datetime": ["date", "time"]},
                infer_datetime_format=True,
                dtype={"freq": np.float32},
                index_col="datetime",
                # nrows= 15 * 60 * 1000
            ))
        self.df = pd.concat(frames)

        # correct offset to ensure even balance throughout the year
        # assume constant measurement offset
        offset = self.df['freq'].mean() - self.F_NOMINAL
        # self.df['freq'] -= offset
        print("Frequency offset: {}".format(offset))

    def load_dtu_data(self):
        self.time_step = 0.5    # data at 0.5 second intervals

        self.df = pd.read_csv(
            self.DTU_DATA,
            header=0,                               # override header names in the file
            names=["datetime", "freq"],
            dtype={"datetime": np.float64, "freq": np.float32},
            # nrows=2 * 15 #* 60 * 1000
        )

        # faster than defining a custom function to parse directly in read_csv
        self.df['datetime'] = pd.to_datetime(self.df['datetime'], unit='ms')
        self.df.set_index('datetime', inplace=True)

        # clean up invalid values, by replacing them with the nominal frequency
        # self.df.dropna(inplace=True)
        self.df.where(self.df.notna(), other=self.F_NOMINAL, inplace=True)

        # correct offset to ensure even balance throughout the year
        # assume constant measurement offset
        offset = self.df['freq'].mean() - self.F_NOMINAL
        self.df['freq'] -= offset

        # resample to 1 second intervals
        self.df = self.df.resample('1S').first()
        self.time_step = 1

    # def compute_delta_f(self):
    #     delta_f = self.df['freq'].to_numpy() - self.F_NOMINAL
    #     self.df.insert(len(self.df.columns), 'delta_f', delta_f)

    def plot_distribution(self):
        # libraries & dataset

        # set default theme
        sns.set_theme()

        # histogram plot with kernel density estimation overlaid
        sns.displot(data=self.df, x="freq", kde=True, bins=200)
        plt.show()

    def get_data_subset(self, duration=None, offset=timedelta(), dt=1):
        start_date = self.df.first_valid_index() + offset
        last_date = self.df.last_valid_index()

        if duration is None:
            end_date = last_date
        else:
            end_date = start_date + duration

        if start_date > last_date:
            raise "Offset too large! The requested start date extends beyond the available data"

        if end_date > last_date:
            warnings.warn("End date exceeds available data. Truncating end date.")
            end_date = last_date

        # Filter to desired date range
        filtered_df = self.df.loc[start_date:end_date]
        return filtered_df.resample("{}S".format(dt)).mean()

    # def plot_energy(self, duration=None, offset=timedelta()):
    #     start_date = self.df.first_valid_index() + offset
    #     last_date = self.df.last_valid_index()
    #
    #     if duration is None:
    #         end_date = last_date
    #     else:
    #         end_date = start_date + duration
    #
    #     if start_date > last_date:
    #         raise "Offset too large! The requested start date extends beyond the available data"
    #
    #     if end_date > last_date:
    #         warnings.warn("End date exceeds available data. Truncating end date.")
    #         end_date = last_date
    #
    #     # Filter to desired date range and resample into 15 minute intervals
    #     df_reduced = self.df['power'].loc[start_date:end_date].resample('15T').sum()
    #     df_cumulative = df_reduced.cumsum()
    #
    #     # df_reduced.plot.line()
    #     df_cumulative.plot.area(stacked=False)
    #     plt.show()


if __name__ == "__main__":
    fd = FrequencyData(FrequencyData.DATA_PATH + '/202*_Frequenz.csv')
    # fd = FrequencyData(FrequencyData.DTU_DATA)
    # fd.plot_distribution()
    # fd.plot_energy(duration=timedelta(days=30), offset=timedelta(days=4))
