import math

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker
from matplotlib import colors, lines
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter, LinearLocator
import seaborn as sns

import StorageSystem

BLUE = '#2CBDFE'
GREEN = '#47DBCD'
PINK = '#F3A0F2'
PURPLE = '#9D2EC5'
VIOLET = '#661D98'
AMBER = '#F5B14C'


def plot_time_curves(ss, save_fig=False, path=None):

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
    axs[n].plot(ss.sim_data['t'], ss.sim_data['batt_soc'], label='SOC', linewidth=0.75)
    axs[n].set_ylabel('SOC [%]')
    axs[n].set_ylim((0, 1))
    axs[n].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    axs[n].hlines([ss.soc_max, ss.soc_min], ss.sim_data['t'][0], ss.sim_data['t'][-1],
                  color=PURPLE, linestyles='dashed', alpha=0.8, label='SOC limits')
    axs[n].hlines([ss.soc_sell_trigger, ss.soc_buy_trigger], ss.sim_data['t'][0], ss.sim_data['t'][-1],
                  color=GREEN, linestyles='dashdot', alpha=0.8, label='Trade triggers')
    axs[n].hlines([ss.soc_target], ss.sim_data['t'][0], ss.sim_data['t'][-1],
                  color=AMBER, linestyles='dotted', alpha=0.8, label='SOC Target')

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

    if save_fig:
        filename = 'time' + get_ss_file_name(ss)
        plt.savefig(path + filename + '.svg', dpi=150, format='svg')
        plt.savefig(path + filename + '.png', dpi=150, format='png')
    else:
        plt.show()


# https://matplotlib.org/stable/gallery/statistics/hist.html?highlight=2d%20hist
def plot_rel_freq_data(ss, save_fig=False, path=None):
    fig, axs = plt.subplots(tight_layout=True, figsize=(10, 8))     # width, height
    fig.suptitle('Battery Statistics: {}'.format(ss))

    axs = [
        plt.subplot(221),
        plt.subplot(222),
        plt.subplot(212)
    ]
    patches = [[], [], []]

    # 1. C_rate vs. Cycle Depth
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

    axs[n].text(0.95, 0.95,
        "Eq. Full Cycles: {:.2f}".format(ss.battery.eq_full_cycle_count),
        horizontalalignment='right', verticalalignment='top', transform=axs[n].transAxes, fontsize=12,
        bbox={'facecolor': 'white', 'pad': 5, 'edgecolor': (0.85, 0.85, 0.85), 'boxstyle': "Round, pad=0.5"}
    )

    # 2. Energy bar chart
    n += 1
    axs[n].set_title('Energy Balance')
    labels = ['Gain (Bought)', 'Loss (Sold)']

    # Energy gained or lost through transactions
    sold, bought = ss.get_total_trans_volume()
    market = np.array([bought, sold])

    # Energy gained (lost) through allowed FCR manipulations
    deadband = np.array([
        ss.energy['deadband'][0],
        -ss.energy['deadband'][1]
    ])
    over_fulfillment = np.array([
        ss.energy['over_fulfillment'][0],
        -ss.energy['over_fulfillment'][1]
    ])

    # System losses
    losses = np.array([0, ss.battery.accumulated_losses])

    # The energy delivered for the defined FCR product
    # Note: positive fcr power -> discharge battery and vice versa
    fcr_power = ss.compute_fcr_power(ss.sim_data['df'])
    dt = (ss.sim_data['t'][1] - ss.sim_data['t'][0]) / np.timedelta64(1, 'h')
    fcr_energy = fcr_power * dt
    net_e = [-np.sum(fcr_energy[np.where(fcr_energy < 0)]), np.sum(fcr_energy[np.where(fcr_energy > 0)])]

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

    axs[n].text(0.5, 0.5,
        "Net: {:.2f} MWh".format((ss.battery.soc-ss.battery.starting_soc) * ss.battery.capacity_nominal),
        horizontalalignment='center', verticalalignment='top', transform=axs[n].transAxes, fontsize=12,
        bbox={'facecolor': 'white', 'pad': 5, 'edgecolor': (0.85, 0.85, 0.85), 'boxstyle': "Round, pad=0.5"}
    )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[n].set_ylabel('Energy MWh')
    axs[n].grid(True, axis='y')
    axs[n].legend(loc='upper center')

    # 3. SOC
    # Use weights to compute relative frequencies
    n += 1
    weights = np.ones_like(ss.sim_data['batt_soc']) / len(ss.sim_data['batt_soc'])
    bin_count, bins, patches[n] = axs[n].hist(ss.sim_data['batt_soc'], bins=25, weights=weights)
    axs[n].xaxis.set_major_formatter(PercentFormatter(xmax=1))
    axs[n].set_ylim((0, 0.1))
    axs[n].set_xlabel('Battery SOC [%]')
    axs[n].vlines([ss.soc_max, ss.soc_min], 0, 1,
                  color=PURPLE, linestyles='dashed', label='SOC limits')
    axs[n].vlines([ss.soc_sell_trigger, ss.soc_buy_trigger], 0, 1,
                  color=GREEN, linestyles='dashdot', label='Trade triggers')
    axs[n].vlines([ss.soc_target], 0, 1,
                  color=AMBER, linestyles='dotted', label='Target SOC')
    axs[n].legend()
    axs[n].grid(True, which='both')
    axs[n].set_ylabel('Relative Frequency')
    axs[n].yaxis.set_major_formatter(PercentFormatter(xmax=1))

    if save_fig:
        filename = 'stats' + get_ss_file_name(ss)
        plt.savefig(path + filename + '.svg', dpi=150, format='svg')
        plt.savefig(path + filename + '.png', dpi=150, format='png')
    else:
        plt.show()


def plot_summary(storage_systems, capacities, save_fig=False, path=None):

    start_date = np.datetime64(storage_systems[0].sim_data['t'][0], 'D')
    end_date = np.datetime64(storage_systems[0].sim_data['t'][-1], 'D')
    p_market = storage_systems[0].p_market
    fig, axs = plt.subplots(tight_layout=True, figsize=(10, 8))     # width, height
    fig.suptitle('Revenue Evaluation, Marketable Power = {:.0f} MWh: {} to {}'.format(p_market, start_date, end_date))

    axs = [
        plt.subplot(221),
        plt.subplot(222),
        plt.subplot(223),
        plt.subplot(224),
    ]

    # Build dataset
    battery_types = {}
    markups_markdowns = np.arange(0, 0.5, 0.2)
    for ss in storage_systems:
        battery_type = ss.battery.name
        if battery_type not in battery_types:
            battery_types[battery_type] = []
        var_financials = []
        for markup_down in markups_markdowns:
            var_financials.append(ss.compute_annual_var_financials(markup=markup_down, markdown=markup_down))
        sys_costs = ss.get_system_cost_annuity()
        battery_types[battery_type].append((ss, sys_costs, var_financials))

    # Generic
    markers = ['v', 'o', '+', 'h', '*', 'H', 'x', 'D']
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    hatches = ['', '//', '.', '*', '-', '+', 'x', 'o', 'O', '\\', '|']
    legend_entries = []
    n_batt_types = len(battery_types)
    for bat_type_index, battery_type in enumerate(battery_types.keys()):
        legend_entries.append(
            Patch(facecolor=colours[bat_type_index], edgecolor=colours[bat_type_index], label=battery_type)
        )

    # 1. Market Revenue (FCR + Intra-day)
    n = 0
    axs[n].set_title('Annualized Market Revenue (FCR + Intraday)\n (with varying intraday markups vs. day-ahead)')
    axs[n].set_xlabel('Battery Capacity [MWh]')
    axs[n].set_ylabel('Market Rev. [k€]')
    axs[n].grid(True)
    axs[n].set_axisbelow(True)

    sub_legend = []
    for bat_type_index, battery_type in enumerate(battery_types.keys()):
        for system, _, fin_results in battery_types[battery_type]:
            for markup_index, fin_result in enumerate(fin_results):
                x = system.battery.capacity_nominal
                y = (fin_result['total']['revenue'] - fin_result['total']['costs']) * 1e-3  # in k€
                axs[n].scatter(x, y, c=colours[bat_type_index], marker=markers[markup_index])

    for index, markup_down in enumerate(markups_markdowns):
        sub_legend.append(lines.Line2D(
            [0], [0], color='k', marker=markers[index], linewidth=0, label="{:.0%} markup".format(markup_down)
        ))
    axs[n].legend(handles=sub_legend)

    # 2. System Costs
    #    Bar chart: Initial investment, additional investment, op & maintenance
    n += 1
    axs[n].set_title('System Costs (20 year annuity)')
    axs[n].set_xlabel('Battery Capacity [MWh]')
    axs[n].set_ylabel('Annuity [k€]')
    sub_legend = []

    capacity_spread = max(capacities) - min(capacities)
    dx = capacity_spread * np.linspace(-1/2, 1/2, n_batt_types, endpoint=True) if n_batt_types > 1 else np.array([0])
    width = 0.5 / n_batt_types / capacity_spread
    for bat_type_index, battery_type in enumerate(battery_types.keys()):
        x1 = []
        y_capex = []
        y_opex = []
        for system, sys_costs, _ in battery_types[battery_type]:
            x1.append(system.battery.capacity_nominal)      # in case not all capacities are available for this type
            y_capex.append((sys_costs['capex']) * 1e-3)     # in k€
            y_opex.append((sys_costs['opex']) * 1e-3)       # in k€
        x1 = np.array(x1)
        axs[n].bar(x1 + dx[bat_type_index], y_capex, width, color=colours[bat_type_index], hatch=hatches[0])
        axs[n].bar(x1 + dx[bat_type_index], y_opex, width,
                   color=colours[bat_type_index], bottom=y_capex, hatch=hatches[1])

    sub_legend.append(Patch(facecolor='w', edgecolor='k', label='CAPEX', hatch=hatches[0]))
    sub_legend.append(Patch(facecolor='w', edgecolor='k', label='OPEX', hatch=hatches[1]))

    axs[n].legend(handles=sub_legend)
    axs[n].grid(True)
    axs[n].set_axisbelow(True)

    # 3. Net Revenue
    #    combine 1 + 2
    n += 1
    markup_index = math.floor(len(markups_markdowns) / 2)
    axs[n].set_title('Net Revenues (markup={:.0%})'.format(markups_markdowns[markup_index]))
    axs[n].set_xlabel('Battery Capacity [MWh]')
    axs[n].set_ylabel('Net Rev. [k€]')

    for bat_type_index, battery_type in enumerate(battery_types.keys()):
        x = []
        y = []
        for system, sys_costs, fin_results in battery_types[battery_type]:
            fin_result = fin_results[markup_index]
            x.append(system.battery.capacity_nominal)
            y.append((fin_result['total']['revenue'] - fin_result['total']['costs'] - sys_costs['total']) * 1e-3)  # k€
        axs[n].plot(x, y, c=colours[bat_type_index], label=battery_type)

    axs[n].grid(True)
    axs[n].set_axisbelow(True)

    # 4. Expected lifetimes
    #    estimated lifespan
    n += 1
    axs[n].set_title('Estimated Battery Lifespan')
    axs[n].set_xlabel('Battery Capacity [MWh]')
    axs[n].set_ylabel('Lifespan [a]')

    for bat_type_index, battery_type in enumerate(battery_types.keys()):
        x = []
        y = []
        for system, _, _ in battery_types[battery_type]:
            year_fraction = system.get_year_fraction()
            est_life = system.battery.estimate_lifespan(year_fraction)
            x.append(system.battery.capacity_nominal)
            y.append(est_life)
        axs[n].plot(x, y, c=colours[bat_type_index], label=battery_type)

    axs[n].grid(True)
    axs[n].set_axisbelow(True)

    fig.legend(handles=legend_entries)

    if save_fig:
        filename = 'summary'
        plt.savefig(path + filename + '.svg', dpi=150, format='svg')
        plt.savefig(path + filename + '.png', dpi=150, format='png')
    else:
        plt.show()


def get_ss_file_name(ss):
    return '_{}_e{:.2f}_tr{:.2f}_nc{:.2f}_nd{:.2f}_nsd{:.2f}_soc{:.2f}_delay{:.2f}'.format(
            ss.battery.name,
            ss.battery.capacity_nominal,
            ss.soc_sell_trigger - 0.5,
            ss.battery.eta_char,
            ss.battery.eta_disc,
            ss.battery.eta_self_disc_s,
            ss.soc_target,
            StorageSystem.TRANSACTION_DELAY
        )


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
        'legend.framealpha': 1,
        'legend.facecolor': 'white',
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
        'ytick.right': False,

    })

    sns.set_context("notebook", rc={
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        'figure.titlesize': "large"
    })

    plt.rcParams['savefig.pad_inches'] = 0.2

    color_list = [BLUE, PINK, GREEN, AMBER, PURPLE, VIOLET]
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


enable_pretty_plots()
