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

    # Nominal frequency [Hz]
    F_NOMINAL = 50

    def __init__(self, data_src):
        self.df = None          # data frame
        self.time_step = 0      # data time step in seconds

        if data_src == self.PQ_DATA:
            self.load_pq_data()
        elif data_src == self.DTU_DATA:
            self.load_dtu_data()
        else:
            raise "Invalid data source provided"

        self.compute_power()

        # diagnostics
        self.df.info(verbose=False, memory_usage="deep")
        print(f'Frequencies range between {self.df["freq"].min():.3f} and {self.df["freq"].max():.3f}')

    def load_pq_data(self):
        self.time_step = 1  # data at 1 second intervals
        self.df = pd.read_csv(
            self.PQ_DATA,
            names=["date", "time", "freq"],
            usecols=[0, 1, 3],
            parse_dates={"datetime": ["date", "time"]},
            infer_datetime_format=True,
            dtype={"freq": np.float32},
            index_col="datetime",
            # nrows= 15 * 60 * 1000
        )

    def load_dtu_data(self):
        self.time_step = 0.5    # data at 0.5 second intervals

        self.df = pd.read_csv(
            self.DTU_DATA,
            header=0,                               # override header names in the file
            names=["datetime", "freq"],
            dtype={"datetime": np.float64, "freq": np.float32},
            # nrows=2 * 15 #* 60 * 1000
        )

        # clean up invalid values
        self.df.dropna(inplace=True)

        # faster than defining a custom function to parse directly in read_csv
        self.df['datetime'] = pd.to_datetime(self.df['datetime'], unit='ms')
        self.df.set_index('datetime', inplace=True)

        # correct offset to ensure even balance throughout the year
        # assume constant measurement offset
        offset = self.df['freq'].mean() - self.F_NOMINAL
        self.df['freq'] -= offset

    def compute_power(self):
        dead_band = 0           # control dead band in Hz
        max_df = 0.2            # maximum frequency deviation for full power [Hz]

        delta_f = (self.df['freq'].to_numpy() - self.F_NOMINAL)

        # apply dead band logic, if using
        delta_f[np.abs(delta_f) <= dead_band] = 0

        # saturate frequency deviation at control maximum
        delta_f = np.clip(delta_f, -max_df, max_df)

        # convert frequency deviation to power [pu for use with hours]
        power = delta_f * self.time_step / 3600

        # TODO account for drift during data period

        self.df.insert(len(self.df.columns), 'power', power)

    def plot_distribution(self):
        # libraries & dataset

        # set default theme
        sns.set_theme()

        # histogram plot with kernel density estimation overlaid
        sns.displot(data=self.df, x="freq", kde=True, bins=200)
        plt.show()

    def plot_energy(self, duration=None, offset=timedelta()):
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

        # Filter to desired date range and resample into 15 minute intervals
        df_reduced = self.df['power'].loc[start_date:end_date].resample('15T').sum()
        df_cumulative = df_reduced.cumsum()

        # df_reduced.plot.line()
        df_cumulative.plot.area(stacked=False)
        plt.show()


if __name__ == "__main__":
    fd = FrequencyData(FrequencyData.DTU_DATA)
   # fd.plot_distribution()
    fd.plot_energy(duration=timedelta(days=30), offset=timedelta(days=4))
