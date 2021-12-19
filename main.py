from datetime import timedelta
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import numpy as np
import seaborn as sns

from Battery import Battery
from FrequencyData import FrequencyData
from StorageSystem import StorageSystem


def simulate_storage(duration):
    fd = FrequencyData(FrequencyData.PQ_DATA)
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

    print("Equivalent full cycle count: {}".format(ss.battery.eq_full_cycle_count))
    print("End SOC: {:.2%}".format(ss.battery.soc))
    print("End Energy Gain: {:.2f} MWh".format((ss.battery.soc-ss.battery.starting_soc) * ss.battery.capacity_nominal))

    plot_time_curves(ss)
    plot_rel_freq_data(ss)


def plot_time_curves(ss):

    n_plots = 5
    fig, axs = plt.subplots(n_plots, 1, constrained_layout=True, figsize=(8, 2*n_plots))
    fig.suptitle('Load Profile: {}'.format(ss))

    # 1. Load Frequency
    n = 0
    axs[n].plot(ss.sim_data['t'], ss.sim_data['freq'], linewidth=0.5)
    axs[n].set_ylabel('Sys Freq [Hz]')

    # 2. FCR Power
    n += 1
    axs[n].plot(ss.sim_data['t'], ss.sim_data['p_fcr'] + ss.sim_data['p_soc_fcr'],
                alpha=1, label='FCR + SOC', linewidth=0.25)
    axs[n].plot(ss.sim_data['t'], ss.sim_data['p_fcr'], label='FCR Only', linewidth=0.25)

    axs[n].set_ylabel('FCR Power [MW]')

    # Put a legend to the right of the current axis
    axs[n].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # 3. Transaction Power
    n += 1
    axs[n].plot(ss.sim_data['t'], ss.sim_data['p_soc_trans'])
    axs[n].set_ylabel('Trans. Power [MW]')
    axs[n].text(1.025, 0.5, "Sold: {:.2f} MWh\nBought: {:.2f} MWh".format(*ss.get_total_trans_volume()),
                horizontalalignment='left', verticalalignment='center', transform=axs[n].transAxes,
                fontsize=12, linespacing=1.5,
                bbox={'facecolor': 'white', 'pad': 5, 'edgecolor': (0.85, 0.85, 0.85), 'boxstyle': "Round, pad=0.5"})

    # 4. Battery Power
    n += 1
    axs[n].plot(ss.sim_data['t'], ss.sim_data['p_batt'], linewidth=0.25)
    axs[n].set_ylabel('Battery Power [MW]')
    axs[n].annotate('Discharging', xy=(1.01, 1), xycoords='axes fraction', ha="left", va="top")
    axs[n].annotate('Charging', xy=(1.01, 0), xycoords='axes fraction', ha="left", va="bottom")

    # 5. SOC
    n += 1
    axs[n].plot(ss.sim_data['t'], ss.sim_data['batt_soc'], label='SOC')
    axs[n].set_ylabel('SOC [%]')
    axs[n].set_ylim((0, 1))
    axs[n].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    axs[n].hlines([ss.soc_max, ss.soc_min], ss.sim_data['t'][0], ss.sim_data['t'][-1],
                  color='#9D2EC5', linestyles='dashed', alpha=0.5, label='SOC limits')
    axs[n].hlines([ss.soc_sell_trigger, ss.soc_buy_trigger], ss.sim_data['t'][0], ss.sim_data['t'][-1],
                  color='#47DBCD', linestyles='dashdot', alpha=0.5, label='Trade triggers')

    # Put a legend to the right of the current axis
    axs[n].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # General adjustments
    for nn, ax in enumerate(axs):

        # format x labels
        major_locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
        minor_locator = mdates.AutoDateLocator(minticks=3*4, maxticks=8*4)
        formatter = mdates.ConciseDateFormatter(major_locator)
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.xaxis.set_major_formatter(formatter)

        axs[nn].grid(True, which='both')
        axs[nn].yaxis.set_label_coords(-0.1, 0.5)

    plt.show()


# https://matplotlib.org/stable/gallery/statistics/hist.html?highlight=2d%20hist
def plot_rel_freq_data(ss):
    fig, axs = plt.subplots(tight_layout=True, figsize=(10, 8))     # width, height
    fig.suptitle('Battery Statistics: '.format(ss))

    axs = [
        plt.subplot(221),
        plt.subplot(222),
        plt.subplot(212)
    ]
    patches = [[], [], []]

    # C_rate vs. Cycle Depth
    # Use weights to compute relative frequencies
    n = 0
    n_bins = 15
    n_pts = len(ss.battery.cycles['c_rate'])
    weights = np.ones(n_pts) / len(ss.battery.cycles['c_rate'])
    h, xedges, yedges, image = axs[n].hist2d(
        x=ss.battery.cycles['c_rate'],
        y=ss.battery.cycles['cycle_depth'],
        bins=n_bins,
        # range=[[xmin, xmax], [ymin, ymax]],
        weights=weights,
        cmap='gradient_map',
        cmin=1/n_pts
    )
    axs[n].set_xlabel('Peak C-rate [1/h]')
    axs[n].set_ylabel('Half-cycle depth [%]')
    axs[n].set_title('Relative Frequency Heat Map')
    axs[n].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    fig.colorbar(image, ax=axs[n], format=PercentFormatter(xmax=1))

    # Energy bar chart
    n += 1
    axs[n].set_title('Energy Balance')
    labels = ['Gain (Bought)', 'Loss (Sold)', 'Net Gain/Loss']
    sold, bought = ss.get_total_trans_volume()
    net_df = np.average(ss.sim_data['df'])
    net_power = -ss.compute_fcr_power(net_df)
    net_energy = net_power * (ss.sim_data['t'][-1] - ss.sim_data['t'][0]) / np.timedelta64(1, 'h')

    market = np.array([bought, sold, 0])
    deadband = np.array([
        ss.energy['deadband'][0],
        -ss.energy['deadband'][1],
        0
    ])
    over_fulfillment = np.array([
        ss.energy['over_fulfillment'][0],
        -ss.energy['over_fulfillment'][1],
        0
    ])
    net_gain = [0, 0, (ss.battery.soc-ss.battery.starting_soc) * ss.battery.capacity_nominal]
    losses = np.array([0, ss.battery.accumulated_losses, 0])
    net_e = np.array([max(0, net_energy), max(0, -1*net_energy), 0])

    width = 0.3  # the width of the bars
    axs[n].bar(labels, market, width, label='Market')
    cum_sum = market
    axs[n].bar(labels, deadband, width, bottom=cum_sum, label='Deadband')
    cum_sum += deadband
    axs[n].bar(labels, over_fulfillment, width, bottom=cum_sum, label='Over-fulfillment')
    cum_sum += over_fulfillment
    axs[n].bar(labels, losses, width, bottom=cum_sum, label='Losses')
    cum_sum += losses
    axs[n].bar(labels, net_e, width, bottom=cum_sum, label='FCR Product')
    cum_sum += net_e
    axs[n].bar(labels, net_gain, width, bottom=cum_sum)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[n].set_ylabel('Energy MWh')
    axs[n].grid(True, axis='y')
    axs[n].legend()

    print("Gain-Loss: {:.2f} MWh".format(cum_sum[0] - cum_sum[1]))

    # SOC
    # Use weights to compute relative frequencies
    n += 1
    weights = np.ones_like(ss.sim_data['batt_soc']) / len(ss.sim_data['batt_soc'])
    bin_count, bins, patches[n] = axs[n].hist(ss.sim_data['batt_soc'], bins=25, weights=weights)
    axs[n].xaxis.set_major_formatter(PercentFormatter(xmax=1))
    axs[n].set_ylim((0, 0.1))
    axs[n].set_xlabel('Battery SOC [%]')
    axs[n].vlines([ss.soc_max, ss.soc_min], 0, 1,
                  color='#47DBCD', linestyles='dashed', alpha=0.5, label='SOC limits')
    axs[n].vlines([ss.soc_sell_trigger, ss.soc_buy_trigger], 0, 1,
                  color='#9D2EC5', linestyles='dashdot', alpha=0.5, label='Trade triggers')
    axs[n].legend()
    axs[n].grid(True, which='both')
    axs[n].set_ylabel('Relative Frequency')
    axs[n].yaxis.set_major_formatter(PercentFormatter(xmax=1))

    plt.show()


def enable_pretty_plots():
    sns.set(font='Franklin Gothic Book', rc={
        'axes.axisbelow': False,
        'axes.edgecolor': 'lightgrey',
        'axes.facecolor': 'None',
        'axes.grid': False,
        'grid.color': (0.85, 0.85, 0.85),
        'grid.linestyle': ':',
        'grid.linewidth': 0.5,
        'axes.labelcolor': 'dimgrey',
        # 'axes.spines.right': False,
        # 'axes.spines.top': False,
        'figure.facecolor': 'white',
        'lines.solid_capstyle': 'round',
        'patch.edgecolor': 'w',
        'patch.force_edgecolor': True,
        'text.color': 'dimgrey',
        # 'xtick.bottom': False,
        'xtick.color': 'dimgrey',
        'xtick.direction': 'out',
        'xtick.top': False,
        'ytick.color': 'dimgrey',
        'ytick.direction': 'out',
        # 'ytick.left': False,
        'ytick.right': False
    })

    sns.set_context("notebook", rc={
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        'figure.titlesize': "large"
    })

    blue = '#2CBDFE'
    green = '#47DBCD'
    pink = '#F3A0F2'
    purple = '#9D2EC5'
    violet = '#661D98'
    amber = '#F5B14C'

    color_list = [blue, pink, green, amber, purple, violet]
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

    gradient_map = ['#2cbdfe', '#2fb9fc', '#33b4fa', '#36b0f8',
                    '#3aacf6', '#3da8f4', '#41a3f2', '#449ff0',
                    '#489bee', '#4b97ec', '#4f92ea', '#528ee8',
                    '#568ae6', '#5986e4', '#5c81e2', '#607de0',
                    '#6379de', '#6775dc', '#6a70da', '#6e6cd8',
                    '#7168d7', '#7564d5', '#785fd3', '#7c5bd1',
                    '#7f57cf', '#8353cd', '#864ecb', '#894ac9',
                    '#8d46c7', '#9042c5', '#943dc3', '#9739c1',
                    '#9b35bf', '#9e31bd', '#a22cbb', '#a528b9',
                    '#a924b7', '#ac20b5', '#b01bb3', '#b317b1']

    plt.colormaps.register(colors.ListedColormap(gradient_map, 'gradient_map'))


if __name__ == "__main__":
    enable_pretty_plots()
    simulate_storage(duration=timedelta(days=1))
