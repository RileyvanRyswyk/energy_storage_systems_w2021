import math
from collections import deque
from datetime import timedelta

import numpy as np
import pandas as pd

from Battery import Battery
from Transaction import Transaction

# TSO parameters
F_NOMINAL = 50              # Nominal frequency [Hz]
DEAD_BAND = 0.01            # allowed control dead band [Hz]
MAX_DELTA_F = 0.2           # maximum frequency deviation for full power [Hz]
MAX_OVER_FULFILLMENT = 0.2  # maximum over fulfillment allowed [%/100]
ALERT_DURATION = 0.5        # duration that full power must be delivered in the alert state [h]

# Related to primary market transactions
TRANSACTION_DURATION = 0.25     # duration of transaction [h]
TRANSACTION_DELAY = 0.5         # delay between placing request and delivery [h]
P_MAX_CONST_DEV = 0.25          # the estimated maximum power required for a sustained deviation [pu] (50 mHz)
FCR_PRODUCT_LENGTH = 4          # length of FCR Product [hours]
DA_PRODUCT_LENGTH = 1           # length of day-ahead product [hours], used to estimate intra-day prices

# Taxes and Fees (per MWh, % price)
# assumed taxes and fees on energy losses, FCR revenues
# current legislation is not entirely clear
# https://www.bmwi.de/Redaktion/EN/Artikel/Energy/electircity-price-components-state-imposed.html
# https://www.bundesnetzagentur.de/SharedDocs/Downloads/DE/Sachgebiete/Energie/Unternehmen_Institutionen/ErneuerbareEnergien/Speicherpapier.pdf?__blob=publicationFile&v=2
# EEG surcharge (6.405 ct/kWh, 2019) -> exempt (2nd, pg. 19)
# CHP surcharge (0.226 ct/kWh, 2019) -> exempt (2nd, pg. 19)
# Grid fee §19  (0.305 ct/kWh, 2019)
# Offshore chr. (0.416 ct/kWh, 2019)
# Interup. load (0.005 ct/kWh, 2019)
# Concession fee -> assume exempt (residential only?)
# electric. tax (2.05 ct/kWh)
# VAT           (19%)
ENERGY_FEES = (0.305 + 0.416 + 0.005 + 2.05) * 1e-2 * 1e3   # € / MWh
ENERGY_TAX = 0.19
FCR_TAX = 0.19

# General:
investment_horizon = 20
capital_costs = 0.05  # % per year

# UPFRONT INVESTMENT:
costs_installed_capacity = 226151  # Euro/MWh (2,75 Euro * 1 MWh / (3,2V * 3,8 Ah))
costs_power_interface = 70000  # Euro/MW
land_acquisition_costs = 0  # Euro
installation_labour_equipment_costs = 70000  # Euro/MWh

# Recurring Costs:
maintenance_and_repair = 0.01  # TODO: Find assumtion (%/MWh)
monitoring_labour_costs = 15000  # Euro/year
supply_energy_costs = 0  # kWh/year

# Aging etc.
recycle_value = 0.1  # %


class StorageSystem:

    def __init__(self, battery=Battery(), p_market=1, p_max=1.25, soc_target=0.5, sell_trigger=1, buy_trigger=0):
        self.battery = battery          # Battery class
        self.p_market = p_market        # marketable power [MW]
        self.p_max = p_max              # maximum power [MW]

        # maximum allowed state of charge in normal state [%/100]
        self.soc_max = (battery.capacity_nominal - ALERT_DURATION * p_market) / battery.capacity_nominal

        # minimum allowed state of charge in normal state [%/100]
        self.soc_min = ALERT_DURATION * p_market / battery.capacity_nominal

        # The amount of energy required to cover the transaction delay period at the maximum assumed constant deviation
        energy_transaction_delay = TRANSACTION_DELAY * P_MAX_CONST_DEV * self.p_market

        # SOC when a sell transaction should be initiated to maintain SOC within nominal limits during normal operation
        max_sell_trigger = self.soc_max - energy_transaction_delay / self.battery.capacity_nominal
        self.soc_sell_trigger = min(max(0.5, sell_trigger), max(0.5, max_sell_trigger))

        # SOC when a buy transaction should be initiated to maintain SOC within nominal limits during normal operation
        min_buy_trigger = self.soc_min + energy_transaction_delay / self.battery.capacity_nominal
        self.soc_buy_trigger = max(min(0.5, buy_trigger), min(0.5, min_buy_trigger))

        # target SOC [%/100]
        self.soc_target = max(min(soc_target, self.soc_sell_trigger), self.soc_buy_trigger)

        self.scheduled_transactions = deque()
        self.transaction_archive = []

        # store last 15 minutes of values (assuming dt = 5)
        # for computing rolling average
        self.df_recent = deque([0] * 3*60, maxlen=3*60)

        # store data during simulation for plotting later
        self.sim_data = {}
        self.sim_index = 0
        self.energy = {
            'deadband': [0, 0],             # pos, neg
            'over_fulfillment': [0, 0]      # pos, neg
        }

        # Market data (FCR & Day-ahead)
        self.market_data = None
        self.fcr_active = None

    def __eq__(self, other):
        if not isinstance(other, StorageSystem):
            raise NotImplementedError

        # examine only float parameters
        for attr in self.__dict__.keys():
            value = getattr(self, attr)
            if type(value) in [float, np.float64]:
                if getattr(other, attr) != value:
                    return False

        return self.battery == other.battery

    def __lt__(self, other):
        if not isinstance(other, StorageSystem):
            raise NotImplementedError

        return self.battery.capacity_nominal < other.battery.capacity_nominal

    def __gt__(self, other):
        if not isinstance(other, StorageSystem):
            raise NotImplementedError

        return self.battery.capacity_nominal > other.battery.capacity_nominal

    def __str__(self):
        dt = self.sim_data['t'][-1] - self.sim_data['t'][0]
        dh = np.round(dt / np.timedelta64(1, 'h'))
        return "{}: {} MW, {} MWh, {} hours, \u03B7c = {:.1%}, \u03B7d = {:.1%}".format(
            self.battery.name, self.p_market, self.battery.capacity_nominal,
            dh, self.battery.eta_char, self.battery.eta_disc
        )

    def link_market_data(self, market_data):
        self.market_data = market_data
        # self.fcr_active = pd.DataFrame(0, index=market_data.fcr_df.index, columns=['active_frac'], dtype=float)
        self.fcr_active = dict.fromkeys(market_data.fcr_df.index, 0.0)

    def init_sim_data(self, n_data_pts):
        self.sim_index = 0
        self.sim_data = {
            't': np.zeros(n_data_pts, dtype='datetime64[s]'),
            'freq': np.zeros(n_data_pts, dtype=np.float32),
            'df': np.zeros(n_data_pts, dtype=np.float32),
            'p_batt': np.zeros(n_data_pts, dtype=np.float32),
            'p_fcr': np.zeros(n_data_pts, dtype=np.float32),
            'p_soc_fcr': np.zeros(n_data_pts, dtype=np.float32),
            'p_soc_trans': np.zeros(n_data_pts, dtype=np.float32),
            'batt_soc': np.zeros(n_data_pts, dtype=np.float32),
        }

    def reset(self):
        self.init_sim_data(self.sim_data['t'].size)
        self.energy['deadband'] = [0, 0]
        self.energy['over_fulfillment'] = [0, 0]
        self.df_recent.clear()
        self.scheduled_transactions.clear()
        self.transaction_archive.clear()
        self.battery.reset()
        self.market_data = None
        self.fcr_active = None

    # Execute one time step of the storage system simulation
    #
    # df : frequency delta in [Hz]
    # t  : current timestamp [datetime]
    # dt : time delta [s]
    def execute_step(self, freq, t, dt=1):
        df = freq - F_NOMINAL
        self.df_recent.append(df)

        # SOC management via transactions
        delta_soc_sch_trans, p_soc_trans = self.manage_soc_trans(t)

        if self.is_fcr_active(t, dt, df):
            # FCR power (product);
            # Power for SOC management via allowed manipulations (dead band, over-fulfillment, activation delay)
            p_fcr, p_soc_fcr = self.compute_net_fcr_power(df, p_soc_trans, delta_soc_sch_trans, dt)
            p_batt = (p_fcr + p_soc_fcr + p_soc_trans)
        else:
            p_batt = p_soc_trans
            p_fcr = 0
            p_soc_fcr = 0

        # Convert to energy units in hours
        e = (p_batt * dt / 3600)
        e_batt_act, p_batt_act = self.battery.execute_step(e, p_batt, dt)

        # store results from this step
        self.sim_data['t'][self.sim_index] = t
        self.sim_data['freq'][self.sim_index] = freq
        self.sim_data['df'][self.sim_index] = df
        self.sim_data['p_batt'][self.sim_index] = p_batt_act
        self.sim_data['p_fcr'][self.sim_index] = p_fcr
        self.sim_data['p_soc_fcr'][self.sim_index] = p_soc_fcr
        self.sim_data['p_soc_trans'][self.sim_index] = p_soc_trans
        self.sim_data['batt_soc'][self.sim_index] = self.battery.soc
        self.sim_index += 1

    # Compute the change in state of charge due to scheduled transactions
    # positive transactions -> sell power
    # negative transactions -> buy power
    def manage_soc_trans(self, t):
        # 1a. find energy of currently schedule transactions
        # 1b. find power of active transactions - should only be one
        energy = 0
        power = 0
        for tran in self.scheduled_transactions:
            remaining_hours = min(tran.end_time - t, tran.end_time - tran.start_time) / timedelta(hours=1)
            energy += remaining_hours * tran.power
            if remaining_hours < 0:
                raise "Transaction not properly archived!"
            if t in tran:
                power += tran.power

        # change in battery SOC based on future transactions
        delta_soc = -energy / self.battery.capacity_nominal

        # 2. At 15 min intervals,
        #   a. archive old transactions
        #   b. check if a new transaction is needed!
        if t == ceil_dt_15min(t):
            self.archive_transactions(t)
            self.create_transaction(t, delta_soc)

        return delta_soc, power

    def is_fcr_active(self, t, dt, df):
        if math.isnan(df):
            return False

        if self.market_data is not None:
            product_start = floor_dt_h(t, FCR_PRODUCT_LENGTH)
            if product_start not in self.fcr_active:
                raise Exception('Missing fcr market data for {}'.format(t))
            self.fcr_active[product_start] += dt / 3600 / FCR_PRODUCT_LENGTH

        return True

    # Compute the change in state of charge due to scheduled transactions
    # positive -> deliver power
    # negative -> consume power
    def compute_net_fcr_power(self, df, p_soc_trans, delta_soc_sch_trans, dt):
        p_fcr = self.compute_fcr_power(df)

        # active energy management via allowed FCR power manipulation
        p_soc = 0
        net_p = p_fcr + p_soc_trans
        soc_error = self.battery.soc + delta_soc_sch_trans - self.soc_target
        if np.abs(net_p) < self.p_max and soc_error != 0 and df != 0:

            # 1. over-fulfillment
            if np.sign(soc_error) * np.sign(p_fcr) > 0:
                p_soc = p_fcr * min(MAX_OVER_FULFILLMENT, (self.p_max - np.abs(net_p)) / self.p_market)
                self.energy['over_fulfillment'][1 * (p_fcr > 0)] -= p_soc * dt / 3600

            # 2. dead band
            elif 0 <= np.abs(df) <= DEAD_BAND:
                p_soc = -p_fcr
                self.energy['deadband'][1 * (p_fcr < 0)] -= p_soc * dt / 3600

        return p_fcr, p_soc

    def compute_fcr_power(self, df):
        # Formula 3.6 of PQ Bedingungen (page 58, 1736)
        if type(df) is float:
            # np.clip was very slow for single floats!
            p = -df / MAX_DELTA_F
            return max(-1, min(1, p)) * self.p_market
        else:
            return -np.clip(df / MAX_DELTA_F, -1, 1) * self.p_market

    def archive_transactions(self, t):
        # find expired transactions
        n_archive = 0
        for tran in self.scheduled_transactions:
            if t >= tran.end_time:
                n_archive += 1
            else:
                # oldest transactions first
                break
        # archive transactions
        for i in range(n_archive):
            self.transaction_archive.append(self.scheduled_transactions.popleft())

    def create_transaction(self, t, delta_soc):
        # TODO transaction size limits

        # est_delta_e = -self.compute_fcr_power(np.average(self.df_recent)) * TRANSACTION_DELAY
        # est_delta_soc = est_delta_e / self.battery.capacity_nominal
        est_delta_soc = 0

        future_soc = self.battery.soc + delta_soc
        est_future_soc = future_soc + est_delta_soc

        if max(future_soc, est_future_soc) > self.soc_sell_trigger:
            self.scheduled_transactions.append(Transaction(
                start_time=t+timedelta(hours=TRANSACTION_DELAY),
                duration=timedelta(hours=TRANSACTION_DURATION),
                power=self.p_max - self.p_market
            ))
        elif min(future_soc, est_future_soc) < self.soc_buy_trigger:
            self.scheduled_transactions.append(Transaction(
                start_time=t+timedelta(hours=TRANSACTION_DELAY),
                duration=timedelta(hours=TRANSACTION_DURATION),
                power=-(self.p_max - self.p_market)
            ))

    def compute_annual_var_financials(self, markup, markdown):
        sold_eur, bought_eur = self.get_trans_revenue(markup, markdown)
        sold_mwh, bought_mwh = self.get_total_trans_volume()
        losses = self.battery.accumulated_losses
        avg_purchase_price = (bought_eur / bought_mwh) if bought_mwh > 0 else 0

        fcr_unit_revenue = 0
        for start_time, active_frac in self.fcr_active.items():
            price = self.market_data.fcr_df.at[start_time, 'price']

            # sometimes there are two tenders, only one produces a valid value
            if type(price) is pd.Series:
                for time_stamp, price_entry in list(price.items()):
                    if type(price_entry) is float:
                        price = price_entry
                        break

            fcr_unit_revenue += active_frac * price

        year_fraction = self.get_year_fraction()

        financials = {
            'revenue': {
                'intraday': self.annualize_and_round(sold_eur, year_fraction),
                'fcr': self.annualize_and_round(self.p_market * fcr_unit_revenue, year_fraction)
            },
            'costs': {
                'intraday': self.annualize_and_round(bought_eur, year_fraction),
                'fees': self.annualize_and_round(ENERGY_FEES * losses, year_fraction),
                'taxes': self.annualize_and_round(ENERGY_TAX * avg_purchase_price * losses, year_fraction)
            },
            'total': {}
        }

        financials['total']['revenue'] = np.sum(list(financials['revenue'].values()))
        financials['total']['costs'] = np.sum(list(financials['costs'].values()))

        return financials

    def get_total_trans_volume(self):
        sold = 0
        bought = 0
        for transaction in self.transaction_archive:
            if transaction.power > 0:
                sold += transaction.power
            else:
                bought += transaction.power
        return sold * TRANSACTION_DURATION, -bought * TRANSACTION_DURATION

    def get_trans_revenue(self, markup, markdown):
        sold = 0
        bought = 0
        for transaction in self.transaction_archive:
            # day ahead data comes in one hour intervals
            da_product_start = floor_dt_h(transaction.start_time, DA_PRODUCT_LENGTH)

            if da_product_start not in self.market_data.da_df.index:
                raise Exception('Incomplete day-ahead data: {}'.format(transaction.start_time))

            price = self.market_data.da_df.at[da_product_start, 'price']

            if transaction.power > 0:
                sold += price * transaction.power * TRANSACTION_DURATION * (1 - markdown)
            else:
                bought -= price * transaction.power * TRANSACTION_DURATION * (1 + markup)
        return sold, bought

    def annualize_and_round(self, value, year_fraction):
        return round(value / year_fraction, 2)

    def get_year_fraction(self):
        dt = self.sim_data['t'][-1] - self.sim_data['t'][0]
        return dt / np.timedelta64(1, 'h') / 8760

    def get_system_cost_annuity(self):
        capacity = self.battery.capacity_nominal

        # cycles_at_defined_dod = 3000 + (capacity/0.25 - 25)*500 #dummy
        # expected_lifetime = int(cycles_at_defined_dod / (cycles_per_day * 365)) # TODO: Better calculation
        storage_lifetime = int(self.battery.estimate_lifespan(self.get_year_fraction()))

        #CAPEX:
        #Points of investments:

        initial_costs = costs_installed_capacity * capacity + \
                        self.p_market * costs_power_interface + \
                        land_acquisition_costs + \
                        installation_labour_equipment_costs

        investment_costs = [initial_costs]

        points_of_reinvestment = []
        for t in range(1, investment_horizon):
            if(np.mod(t, storage_lifetime) == 0):
                points_of_reinvestment.append(t)

        #NPV of initial and replacement investments:
        npv_investment = 0
        for i in points_of_reinvestment:
            npv_investment += (costs_installed_capacity * capacity + self.p_market * costs_power_interface) * \
                                      np.power(1 - capital_costs, i)

            investment_costs.append((costs_installed_capacity * capacity + costs_power_interface * self.p_market))

        if (1 - (investment_horizon - np.max(points_of_reinvestment)) / storage_lifetime > recycle_value):
            linear_depreciation_factor = 1 - (investment_horizon - np.max(points_of_reinvestment)) / storage_lifetime
        else:
            linear_depreciation_factor = recycle_value

        npv_salvage_value = (linear_depreciation_factor * (costs_power_interface * self.p_market + costs_installed_capacity * capacity) \
                    + land_acquisition_costs) * np.power(1 - capital_costs, investment_horizon)

        npv_capex = npv_investment - npv_salvage_value

        #OPEX:
        yearly_expenses_maintenance = []
        yearly_expenses_labour = []
        yearly_expenses_supply_costs = []
        for t in range(investment_horizon):
            #Sauer Lecture OPEX Calculation - Maintenance
            yearly_expenses_maintenance.append(np.power((1-capital_costs),t) * \
            maintenance_and_repair * capacity * costs_installed_capacity)
            #Labour and energy supply costs
            yearly_expenses_labour.append(np.power((1-capital_costs),t) * monitoring_labour_costs)
            yearly_expenses_supply_costs.append(np.power((1-capital_costs),t) * supply_energy_costs)

        npv_opex = np.sum(yearly_expenses_maintenance + yearly_expenses_labour + yearly_expenses_supply_costs)

        # annuity factor
        a = math.pow(1 + capital_costs, investment_horizon)
        annuity_factor = capital_costs * a / (a - 1)

        return {
            'capex':    npv_capex * annuity_factor,
            'opex':     npv_opex * annuity_factor,
            'total':    (npv_capex + npv_opex) * annuity_factor
        }


# https://stackoverflow.com/questions/13071384/ceil-a-datetime-to-next-quarter-of-an-hour
# round datetime to next 15 mins
def ceil_dt_15min(dt):
    # how many secs have passed this hour
    nsecs = dt.minute * 60 + dt.second
    # number of seconds to next quarter hour mark
    delta = math.ceil(nsecs / 900) * 900 - nsecs
    # time + number of seconds to quarter hour mark.
    return dt + timedelta(seconds=delta)


def ceil_dt_h(dt, num_hours):
    # how many secs have passed since last cutoff
    nsecs = (dt.hour % num_hours) * 3600 + dt.minute * 60 + dt.second
    # number of seconds in period
    nsecs_period = 3600 * num_hours
    # number of seconds to next cutoff
    delta = math.ceil(nsecs / nsecs_period) * nsecs_period - nsecs
    # time + number of seconds to cutoff.
    return dt + timedelta(seconds=delta)


def floor_dt_h(dt, num_hours):
    # how many secs have passed since last cutoff
    delta = (dt.hour % num_hours) * 3600 + dt.minute * 60 + dt.second
    return dt - timedelta(seconds=delta)
