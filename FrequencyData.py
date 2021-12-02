import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class FrequencyData:

    def __init__(self, csv_path):
        self.df = pd.read_csv(
            csv_path,
            names=["date", "time", "freq"],
            usecols=[0, 1, 3],
            parse_dates={"datetime": ["date", "time"]},
            infer_datetime_format=True,
            dtype={"freq": np.float32},
            index_col="datetime",
            #nrows= 15 * 60 * 1000
        )
        self.df.info(verbose=False, memory_usage="deep")

        print(f'Frequencies range between {self.df["freq"].min():.3f} and {self.df["freq"].max():.3f}')

    def plot_distribution(self):
        # libraries & dataset

        # set default theme
        sns.set_theme()

        # histogram plot with kernel density estimation overlaid
        sns.displot(data=self.df, x="freq", kde=True, bins=200)
        plt.show()

    def plot_energy(self):

        # resolution in minutes
        nominal_freq = 50
        max_d_f = 0.2

        # TODO saturate results for case with |delta_f| > 0.2

        # convert frequency deviation to energy required in Wh for a 1 W device
        df_energy = (self.df - nominal_freq) / max_d_f / 3600
        df_reduced = df_energy.resample('15T').sum()
        df_cumulative = df_reduced.cumsum()

        df_reduced.plot.line()
        df_cumulative.plot.area(stacked=False)
        plt.show()


if __name__ == "__main__":
    fd = FrequencyData("data/201901_Frequenz.csv")
    #fd.plot_distribution()
    fd.plot_energy()
