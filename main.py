from datetime import timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from Battery import Battery
from FrequencyData import FrequencyData
from StorageSystem import StorageSystem


def simulate_day():
    fd = FrequencyData(FrequencyData.PQ_DATA)
    data = fd.get_data_subset(duration=timedelta(days=1))

    battery = Battery(eta_char=0.95, eta_disc=0.95)
    ss = StorageSystem(battery)

    # add empty rows for simulation results
    i = 0
    results = np.zeros((len(data), 3))
    for row in data.itertuples(index=False):
        results[i] = ss.execute_step(df=row.delta_f, dt=fd.time_step)
        i += 1

    # plot
    fig, ax1 = plt.subplots()

    # df
    ax1.plot(data.index, data['delta_f'], linewidth=1.0, alpha=0.5)

    # Power
    ax1.plot(data.index, results[:, 0], linewidth=1.0, alpha=0.5)

    # format x labels
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)

    ax1.set_xlabel('Date/Time')
    ax1.set_ylabel('Charge/Discharge Power [pu] / Δf [Hz]')

    ax2 = ax1.twinx()

    # SOC - convert to percent
    ax2.plot(data.index, 100 * results[:, 2], linewidth=1.0)
    ax2.set_ylabel('SOC [%]')
    ax2.set_ylim(bottom=0, top=100)

    plt.show()


if __name__ == "__main__":
    simulate_day()