from datetime import timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from Battery import Battery
from Dataset import Dataset
from FrequencyData import FrequencyData
from StorageSystem import StorageSystem



def simulate_day():
    fd = FrequencyData(FrequencyData.PQ_DATA)
    data = fd.get_data_subset(duration=timedelta(days=1))

    battery = Battery(eta_char=0.95, eta_disc=0.95)
    ss = StorageSystem(battery)

    # add empty rows for simulation results
    i = 0
    results = np.zeros((len(data), 5))
    for row in data.itertuples(index=True):
        results[i] = ss.execute_step(df=row.delta_f, dt=fd.time_step, t=row.Index)
        i += 1

    # p_batt, p_fcr, p_soc_fcr, p_soc_trans, self.battery.soc
    plot_data = [Dataset(
        x=data.index,
        y=data['freq'],
        title='Daily Load Profile',
        ylabel=' Sys Freq [Hz]'
    ), Dataset(
        x=data.index,
        y=results[:, 1],
        ylabel='FCR Power [pu]'
    ), Dataset(
        x=data.index,
        y=results[:, 2],
        ylabel='FCR SOC Power [pu]'
    ), Dataset(
        x=data.index,
        y=results[:, 3],
        ylabel='Trans. Power [pu]'
    ), Dataset(
        x=data.index,
        y=results[:, 0],
        ylabel='Battery Power [pu]'
    ), Dataset(
        x=data.index,
        y=100 * results[:, 4],
        ylabel='SOC [%]',
        ylim=(0, 100)
    )]

    plot(plot_data)


def plot(plot_data):
    n_plots = len(plot_data)
    fig, axs = plt.subplots(n_plots, 1, constrained_layout=True, figsize=(8, 2*n_plots))

    for nn, ax in enumerate(axs):
        ax.plot(plot_data[nn].x, plot_data[nn].y)

        # format x labels
        locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        if plot_data[nn].title is not None:
            ax.set_title(plot_data[nn].title)

        if plot_data[nn].xlabel is not None:
            ax.set_xlabel(plot_data[nn].xlabel)
        if plot_data[nn].ylabel is not None:
            ax.set_ylabel(plot_data[nn].ylabel)

        if plot_data[nn].xlim is not None:
            ax.set_xlim(plot_data[nn].xlim)
        if plot_data[nn].ylim is not None:
            ax.set_ylim(plot_data[nn].ylim)

        ax.grid(True, alpha=0.3)

    plt.show()

if __name__ == "__main__":
    simulate_day()