from datetime import timedelta
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import numpy as np

from Battery import Battery
from FrequencyData import FrequencyData
from StorageSystem import StorageSystem


def simulate_storage(duration):
    fd = FrequencyData(FrequencyData.DTU_DATA)
    data = fd.get_data_subset(duration=duration)

    # https://www.mdpi.com/2313-0105/2/3/29/pdf section 5.1
    # from Energy Neighbour project, self discharge is negligible during constant operation
    # losses primarily from power electronics, auxillary equipment
    battery = Battery(eta_char=0.90, eta_disc=0.90, eta_self_disc=0, capacity_nominal=7.5)
    ss = StorageSystem(battery, p_market=5, p_max=6.25)

    # add empty rows for simulation results
    ss.init_sim_data(len(data))
    current_day = None
    for row in data.itertuples(index=True):
        ss.execute_step(freq=row.freq, dt=fd.time_step, t=row.Index)
        if current_day is None or current_day != row.Index.date().day:
            current_day = row.Index.date().day
            print("Currently evaluating {}".format(row.Index.date()))

    plot_time_curves(ss)
    plot_rel_freq_data(ss)


def plot_time_curves(ss):

    n_plots = 5
    fig, axs = plt.subplots(n_plots, 1, constrained_layout=True, figsize=(8, 2*n_plots))

    # 1. Load Frequency
    n = 0
    axs[n].plot(ss.sim_data['t'], ss.sim_data['freq'])
    axs[n].set_title('Load Profile')
    axs[n].set_ylabel('Sys Freq [Hz]')

    # 2. FCR Power
    n += 1
    axs[n].plot(ss.sim_data['t'], ss.sim_data['p_fcr'], label='FCR Only')
    axs[n].plot(ss.sim_data['t'], ss.sim_data['p_fcr'] + ss.sim_data['p_soc_fcr'], alpha=0.5, label='FCR + SOC')
    axs[n].legend(loc='upper right')
    axs[n].set_ylabel('FCR Power [MW]')

    # 3. Transaction Power
    n += 1
    axs[n].plot(ss.sim_data['t'], ss.sim_data['p_soc_trans'])
    axs[n].set_ylabel('Trans. Power [MW]')
    axs[n].text(0.99, 0.08, "Sold: {:.2f} MWh\n Bought: {:.2f} MWh".format(*ss.get_total_trans_volume()),
                verticalalignment='bottom', horizontalalignment='right',
                transform=axs[n].transAxes, fontsize=10,
                bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})

    # 4. Battery Power
    n += 1
    axs[n].plot(ss.sim_data['t'], ss.sim_data['p_batt'])
    axs[n].set_ylabel('Battery Power [MW]')

    # 5. SOC
    n += 1
    axs[n].plot(ss.sim_data['t'], ss.sim_data['batt_soc'], label='SOC')
    axs[n].set_ylabel('SOC [%]')
    axs[n].set_ylim((0, 1))
    axs[n].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    axs[n].hlines([ss.soc_max, ss.soc_min], ss.sim_data['t'][0], ss.sim_data['t'][-1],
                  color='r', linestyles='dashed', alpha=0.5, label='SOC limits')
    axs[n].hlines([ss.soc_sell_trigger, ss.soc_buy_trigger], ss.sim_data['t'][0], ss.sim_data['t'][-1],
                  color='r', linestyles='dashdot', alpha=0.5, label='Trade triggers')
    axs[n].legend(loc='upper right')

    # General adjustments
    for nn, ax in enumerate(axs):

        # format x labels
        major_locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
        minor_locator = mdates.AutoDateLocator(minticks=3*4, maxticks=8*4)
        formatter = mdates.ConciseDateFormatter(major_locator)
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.xaxis.set_major_formatter(formatter)

        axs[nn].grid(True, which='both', alpha=0.3)

    plt.show()


# https://matplotlib.org/stable/gallery/statistics/hist.html?highlight=2d%20hist
def plot_rel_freq_data(ss):
    print("Equivalent full cycle count: {}".format(ss.battery.eq_full_cycle_count))

    n_bins = 50
    fig, axs = plt.subplots(tight_layout=True, figsize=(8, 8))
    fig.suptitle('Battery Statistics')

    axs = [
        plt.subplot(221),
        plt.subplot(222),
        plt.subplot(212)
    ]
    patches = [[], [], []]

    # C_rate
    # Use weights to compute relative frequencies
    n = 0
    weights = np.ones_like(ss.battery.cycles['c_rate']) / len(ss.battery.cycles['c_rate'])
    bin_count, bins, patches[n] = axs[n].hist(ss.battery.cycles['c_rate'], bins=n_bins, weights=weights)
    axs[n].set_ylim((0, 0.2))
    axs[n].set_xlabel('Peak C-rate [1/h]')

    # Cycle Depth
    # Use weights to compute relative frequencies
    n += 1
    weights = np.ones_like(ss.battery.cycles['cycle_depth']) / len(ss.battery.cycles['cycle_depth'])
    bin_count, bins, patches[n] = axs[n].hist(ss.battery.cycles['cycle_depth'], bins=n_bins, weights=weights)
    axs[n].xaxis.set_major_formatter(PercentFormatter(xmax=1))
    axs[n].set_ylim((0, 0.1))
    axs[n].set_xlabel('Half-cycle depth [%]')

    # SOC
    # Use weights to compute relative frequencies
    n += 1
    weights = np.ones_like(ss.sim_data['batt_soc']) / len(ss.sim_data['batt_soc'])
    bin_count, bins, patches[n] = axs[n].hist(ss.sim_data['batt_soc'], bins=n_bins, weights=weights)
    axs[n].xaxis.set_major_formatter(PercentFormatter(xmax=1))
    axs[n].set_ylim((0, 0.1))
    axs[n].set_xlabel('Battery SOC [%]')
    axs[n].vlines([ss.soc_max, ss.soc_min], 0, 1,
                  color='r', linestyles='dashed', alpha=0.5, label='SOC limits')
    axs[n].vlines([ss.soc_sell_trigger, ss.soc_buy_trigger], 0, 1,
                  color='r', linestyles='dashdot', alpha=0.5, label='Trade triggers')
    axs[n].legend(loc='upper right')

    # General adjustments
    for nn, ax in enumerate(axs):
        axs[nn].set_ylabel('Relative Frequency')
        axs[nn].yaxis.set_major_formatter(PercentFormatter(xmax=1))
        axs[nn].grid(True, which='both', alpha=0.3)

        for patch in patches[nn]:
            patch.set(alpha=0.5, edgecolor='k')

    plt.show()

# def plot_cycle_data(ss):
#     print("Equivalent full cycle count: {}".format(ss.battery.eq_full_cycle_count))
#
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     x_min = -0.5
#     x_max = 0.5
#     y_min = 0
#     y_max = 0.1
#     n_bins = 20
#     hist, xedges, yedges = np.histogram2d(ss.battery.cycles['c_rate'], ss.battery.cycles['cycle_depth'],
#                                           bins=n_bins,
#                                           range=[[x_min, x_max], [y_min, y_max]],
#                                           density=True)
#
#     # Anchor positions
#     xpos, ypos = np.meshgrid(xedges[:-1] + (np.abs(x_min)+np.abs(x_max))/n_bins,
#                              yedges[:-1] + (np.abs(y_min)+np.abs(y_max))/n_bins,
#                              indexing="ij")
#     xpos = xpos.ravel()
#     ypos = ypos.ravel()
#     zpos = 0
#
#     # Construct arrays with the dimensions for the 16 bars.
#     dx = (np.abs(x_min)+np.abs(x_max))/n_bins * np.ones_like(zpos)
#     dy = (np.abs(y_min)+np.abs(y_max))/n_bins * np.ones_like(zpos)
#     dz = hist.ravel()
#
#     ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
#
#     plt.show()

if __name__ == "__main__":
    simulate_storage(duration=timedelta(days=30))
