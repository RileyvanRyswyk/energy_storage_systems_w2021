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

# Interest rate WACC
COST_OF_CAPITAL = 0.05

# General:
investment_horizon = 20*12  # months
capital_costs = COST_OF_CAPITAL / 12  # % per month

# UPFRONT INVESTMENT:
# Onetime:
costs_grid_integration = 27000  # Euro/MW
costs_project_development = 70000   # Euro/MWh
property_costs = 1000000    # Euro
costs_controls_and_communication = 24000   # Euro/MW
# (Re)investment:
costs_construction_and_equipment = 58000    # Euro/MWh
costs_system_integration = 48500    # Euro/Mwh
costs_power_equipment = 70000   # Euro/MW
costs_installed_capacity = 226151   # Euro/MWh (2,75 Euro * 1 MWh / (3,2V * 3,8 Ah))

# Recurring Costs:
costs_fixed_o_m = 0.0043/12  # %/month
monitoring_labour_costs = 15000/12  # Euro/month

# Aging etc.
end_of_life_price = 0.1     # resell price after end of lifetime


class StorageSystem:

    def __init__(self, battery=Battery(), p_market=1, p_max=1.25, soc_target=0.5, sell_trigger=1, buy_trigger=0,
                 log_decimation=1, validation_data=None):
        self.battery = battery          # Battery class
        self.p_market = p_market        # marketable power [MW]
        self.p_max = p_max              # maximum power [MW]
        self.soc_max = 1                # maximum allowed state of charge in normal state [%/100]
        self.soc_min = 0                # minimum allowed state of charge in normal state [%/100]
        self.soc_target = soc_target    # target SOC [%/100]

        # SOC when a sell transaction should be initiated to maintain SOC within nominal limits during normal operation
        self.soc_sell_trigger = sell_trigger

        # SOC when a buy transaction should be initiated to maintain SOC within nominal limits during normal operation
        self.soc_buy_trigger = buy_trigger

        # Validation points used to compute soc limits/triggers as the battery capacity fades
        self.soc_validation_data = validation_data

        self.scheduled_transactions = deque()
        self.transaction_archive = []

        # store last 15 minutes of values (assuming dt = 5)
        # for computing rolling average
        self.df_recent = deque([0] * 3*60, maxlen=3*60)

        # store data during simulation for plotting later
        self.log_decimation = log_decimation
        self.sim_data = {}
        self.sim_index = [0, 0]             # undecimated, decimated simulation index
        self.energy = {}
        self.soc_limit_data = {}

        # Market data (FCR & Day-ahead)
        self.market_data = None
        self.fcr_active = None

        # end-of-life
        self.eol_reached = False

        # compute appropriate limits, triggers
        self.compute_soc_limits()

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
        dd = np.round(dt / np.timedelta64(1, 'D'))
        dd_active = self.get_active_time() / 24
        return "{}: {} MW, {} MWh, {:.0f}/{:.0f} Days, \u03B7c = {:.1%}, \u03B7d = {:.1%}".format(
            self.battery.name, self.p_market, self.battery.capacity_nominal,
            dd_active, dd, self.battery.eta_char, self.battery.eta_disc
        )

    def add_soc_validation_data(self, data):
        self.soc_validation_data = data

    def compute_soc_limits(self):
        self.soc_max = (self.battery.capacity_actual - ALERT_DURATION * self.p_market) / self.battery.capacity_actual
        self.soc_min = ALERT_DURATION * self.p_market / self.battery.capacity_actual

        # The amount of energy required to cover the transaction delay period at the maximum assumed constant deviation
        energy_transaction_delay = TRANSACTION_DELAY * P_MAX_CONST_DEV * self.p_market

        validated_buy_trigger, validated_sell_trigger, end_of_life = self.get_soc_validation_data_triggers()

        max_sell_trigger = self.soc_max - energy_transaction_delay / self.battery.capacity_actual
        self.soc_sell_trigger = min(max(0.5, self.soc_sell_trigger), max(0.5, max_sell_trigger), validated_sell_trigger)

        min_buy_trigger = self.soc_min + energy_transaction_delay / self.battery.capacity_actual
        self.soc_buy_trigger = max(min(0.5, self.soc_buy_trigger), min(0.5, min_buy_trigger), validated_buy_trigger)

        self.soc_target = max(min(self.soc_target, self.soc_sell_trigger), self.soc_buy_trigger)

        if len(self.soc_limit_data) > 0:
            self.soc_limit_data['valid_until'].append(self.sim_data['t'][max(0,self.sim_index[1]-1)])
            self.soc_limit_data['soc_max'].append(self.soc_max)
            self.soc_limit_data['soc_min'].append(self.soc_min)
            self.soc_limit_data['buy_soc'].append(self.soc_buy_trigger)
            self.soc_limit_data['sell_soc'].append(self.soc_sell_trigger)

        self.eol_reached = end_of_life
        return end_of_life

    def get_soc_validation_data_triggers(self):
        if self.soc_validation_data is None:
            return math.nan, math.nan, False

        if str(self.battery.capacity_actual) in self.soc_validation_data:
            return self.soc_validation_data[str(self.battery.capacity_actual)]['buy'], \
                   self.soc_validation_data[str(self.battery.capacity_actual)]['sell'], False

        # find capacity data point below and above current value
        capacities = []
        lower = 0
        upper = math.inf
        for capacity in self.soc_validation_data:
            capacity = float(capacity)
            capacities.append(capacity)
            if lower < capacity <= self.battery.capacity_actual:
                lower = capacity
            elif self.battery.capacity_actual <= capacity < upper:
                upper = capacity

        capacities.sort()
        if lower == 0:
            # below lowest data point -> extrapolate
            lower = capacities[0]
            upper = capacities[1]

        buy_trigger = get_linear_interpolation(
            (lower, self.soc_validation_data[str(lower)]['buy']),
            (upper, self.soc_validation_data[str(upper)]['buy']),
            self.battery.capacity_actual
        )

        sell_trigger = get_linear_interpolation(
            (lower, self.soc_validation_data[str(lower)]['sell']),
            (upper, self.soc_validation_data[str(upper)]['sell']),
            self.battery.capacity_actual
        )

        end_of_life = False
        if self.battery.capacity_actual < (2 * capacities[0] - capacities[1]) \
                or buy_trigger > 0.5 or sell_trigger < 0.5:
            end_of_life = True
        elif self.get_year_fraction() > self.battery.CALENDER_LIFESPAN:
            end_of_life = True

        return buy_trigger, sell_trigger, end_of_life

    def link_market_data(self, market_data):
        self.market_data = market_data
        # self.fcr_active = pd.DataFrame(0, index=market_data.fcr_df.index, columns=['active_frac'], dtype=float)
        self.fcr_active = dict.fromkeys(market_data.fcr_df.index, 0.0)

    def init_sim_data(self, n_data_pts, extend=False):
        if n_data_pts is None:
            self.sim_data = {}
            return

        n_tot = n_data_pts + self.sim_index[0]
        n_pts = math.ceil(n_tot/self.log_decimation) - self.sim_index[1]
        sim_data = {
            't': np.zeros(n_pts, dtype='datetime64[s]'),
            'freq': np.zeros(n_pts, dtype=np.float32),
            'df': np.zeros(n_pts, dtype=np.float32),
            'p_batt': np.zeros(n_pts, dtype=np.float32),
            'p_fcr': np.zeros(n_pts, dtype=np.float32),
            'p_soc_fcr': np.zeros(n_pts, dtype=np.float32),
            'p_soc_trans': np.zeros(n_pts, dtype=np.float32),
            'batt_soc': np.zeros(n_pts, dtype=np.float32),
            'batt_cap': np.zeros(n_pts, dtype=np.float32),

        }

        if extend and len(self.sim_data) > 0:
            for key, arr in sim_data.items():
                self.sim_data[key] = np.concatenate((self.sim_data[key], sim_data[key]), axis=None)
        else:
            self.sim_index = [0, 0]
            self.sim_data = sim_data
            self.energy = {
                'deadband': [0, 0],             # pos, neg
                'over_fulfillment': [0, 0],     # pos, neg
                'fcr_gross': [0, 0]             # pos, neg
            }
            self.soc_limit_data = {
                'valid_until': [],
                'soc_max': [self.soc_max],
                'soc_min': [self.soc_min],
                'buy_soc': [self.soc_buy_trigger],
                'sell_soc': [self.soc_sell_trigger],
            }

    # remove zeroes at end of sim data
    def trim_sim_data(self):
        for key, arr in self.sim_data.items():
            self.sim_data[key] = self.sim_data[key][0:self.sim_index[1]]

    def reset(self):
        self.init_sim_data(None)
        self.df_recent.clear()
        self.scheduled_transactions.clear()
        self.transaction_archive.clear()
        self.battery.reset()
        self.market_data = None
        self.fcr_active = None
        self.eol_reached = False

    # Execute one time step of the storage system simulation
    #
    # df : frequency delta in [Hz]
    # t  : current timestamp [datetime]
    # dt : time delta [s]
    def execute_step(self, freq, t, t_star, dt=1):
        df = freq - F_NOMINAL
        self.df_recent.append(df)

        # SOC management via transactions
        delta_soc_sch_trans, p_soc_trans = self.manage_soc_trans(t, t_star)

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

        # Store data at each decimated simulation index

        if self.sim_index[0] % self.log_decimation == 0:
            self.sim_data['t'][self.sim_index[1]] = t_star
            self.sim_data['freq'][self.sim_index[1]] = freq
            self.sim_data['df'][self.sim_index[1]] = df
            self.sim_data['p_batt'][self.sim_index[1]] = p_batt_act
            self.sim_data['p_fcr'][self.sim_index[1]] = p_fcr
            self.sim_data['p_soc_fcr'][self.sim_index[1]] = p_soc_fcr
            self.sim_data['p_soc_trans'][self.sim_index[1]] = p_soc_trans
            self.sim_data['batt_soc'][self.sim_index[1]] = self.battery.soc
            self.sim_data['batt_cap'][self.sim_index[1]] = self.battery.capacity_actual
            self.sim_index[1] += 1
        self.sim_index[0] += 1

    # Compute the change in state of charge due to scheduled transactions
    # positive transactions -> sell power
    # negative transactions -> buy power
    def manage_soc_trans(self, t, t_star):
        # 1a. find energy of currently schedule transactions
        # 1b. find power of active transactions - should only be one
        energy = 0
        power = 0
        for tran in self.scheduled_transactions:
            remaining_hours = min(tran.end_time - t_star, tran.end_time - tran.start_time) / timedelta(hours=1)
            energy += remaining_hours * tran.power
            if remaining_hours < 0:
                raise "Transaction not properly archived!"
            if t_star in tran:
                power += tran.power

        # change in battery SOC based on future transactions
        delta_soc = -energy / self.battery.capacity_actual

        # 2. At 15 min intervals,
        #   a. archive old transactions
        #   b. check if a new transaction is needed!
        if t_star == ceil_dt_min(t_star, 15):
            self.archive_transactions(t_star)
            self.create_transaction(t, t_star, delta_soc)

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

    def get_active_time(self):
        if self.market_data is not None:
            return sum(list(self.fcr_active.values())) * FCR_PRODUCT_LENGTH
        else:
            dt = self.sim_data['t'][-1] - self.sim_data['t'][0]
            return np.round(dt / np.timedelta64(1, 'h'))

    # Compute the change in state of charge due to scheduled transactions
    # positive -> deliver power
    # negative -> consume power
    def compute_net_fcr_power(self, df, p_soc_trans, delta_soc_sch_trans, dt):
        p_fcr = self.compute_fcr_power(df)
        self.energy['fcr_gross'][1 * (p_fcr > 0)] -= p_fcr * dt / 3600

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

    def create_transaction(self, t, t_star, delta_soc):
        # TODO transaction size limits

        # est_delta_e = -self.compute_fcr_power(np.average(self.df_recent)) * TRANSACTION_DELAY
        # est_delta_soc = est_delta_e / self.battery.capacity_nominal
        est_delta_soc = 0

        future_soc = self.battery.soc + delta_soc
        est_future_soc = future_soc + est_delta_soc

        if max(future_soc, est_future_soc) > self.soc_sell_trigger:
            self.scheduled_transactions.append(Transaction(
                base_start_time=t+timedelta(hours=TRANSACTION_DELAY),
                start_time=t_star+timedelta(hours=TRANSACTION_DELAY),
                duration=timedelta(hours=TRANSACTION_DURATION),
                power=self.p_max - self.p_market
            ))
        elif min(future_soc, est_future_soc) < self.soc_buy_trigger:
            self.scheduled_transactions.append(Transaction(
                base_start_time=t + timedelta(hours=TRANSACTION_DELAY),
                start_time=t_star + timedelta(hours=TRANSACTION_DELAY),
                duration=timedelta(hours=TRANSACTION_DURATION),
                power=-(self.p_max - self.p_market)
            ))

    # return the annualized revenues and costs (as an annuity)
    def compute_annual_var_financials(self, markup, markdown):
        year_fraction = self.get_year_fraction()
        sold_eur_ann, bought_eur_ann = self.get_trans_revenue(markup, markdown, year_fraction)
        sold_mwh, bought_mwh = self.get_total_trans_volume()
        losses = self.battery.accumulated_losses
        avg_purchase_price = (bought_eur_ann * year_fraction / bought_mwh) if bought_mwh > 0 else 0

        # same every year for each system
        # no need to convert to npv and back to annuity
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

        financials = {
            'revenue': {
                'intraday': round(sold_eur_ann, 2),
                'fcr': annualize_and_round(self.p_market * fcr_unit_revenue, year_fraction)
            },
            'costs': {
                'intraday': round(bought_eur_ann, 2),
                'fees': annualize_and_round(ENERGY_FEES * losses, year_fraction),
                'taxes': annualize_and_round(ENERGY_TAX * avg_purchase_price * losses, year_fraction)
            },
            'total': {}
        }

        financials['total']['revenue'] = np.sum(list(financials['revenue'].values()))
        financials['total']['costs'] = np.sum(list(financials['costs'].values()))

        return financials

    # total sold and bought volumes on the intra-day market (in MWh)
    def get_total_trans_volume(self):
        sold = 0
        bought = 0
        for transaction in self.transaction_archive:
            if transaction.power > 0:
                sold += transaction.power
            else:
                bought += transaction.power
        return sold * TRANSACTION_DURATION, -bought * TRANSACTION_DURATION

    # amount of revenue, costs from intra-day market transactions
    # equated into an annuity for multi-year simulations
    def get_trans_revenue(self, markup, markdown, year_fraction):
        sold = {}
        bought = {}
        for transaction in self.transaction_archive:
            # day ahead data comes in one hour intervals
            da_product_start = floor_dt_h(transaction.base_start_time, DA_PRODUCT_LENGTH)

            if da_product_start not in self.market_data.da_df.index:
                raise Exception('Incomplete day-ahead data: {}'.format(transaction.start_time))

            price = self.market_data.da_df.at[da_product_start, 'price']
            year = transaction.start_time.year
            month = transaction.start_time.month - 1

            if transaction.power > 0:
                if year not in sold:
                    sold[year] = np.zeros(12, dtype=np.float32)
                sold[year][month] += price * transaction.power * TRANSACTION_DURATION * (1 - markdown)
            else:
                if year not in bought:
                    bought[year] = np.zeros(12, dtype=np.float32)
                bought[year][month] -= price * transaction.power * TRANSACTION_DURATION * (1 + markup)

        return annuitize_yearly_sums(sold, year_fraction), annuitize_yearly_sums(bought, year_fraction)

    # Fraction of the year simulated
    # in the case of multi-year simulations, result is greater than 1
    def get_year_fraction(self):
        dt = self.sim_data['t'][max(0, self.sim_index[1]-1)] - self.sim_data['t'][0]
        return dt / np.timedelta64(1, 'h') / 8760

    # estimated system lifespan in years
    def estimate_life_span(self):
        if self.eol_reached:
            return self.get_year_fraction()
        else:
            return self.battery.estimate_lifespan(self.get_year_fraction())

    def get_system_cost_annuity(self):
        capacity = self.battery.capacity_nominal
        storage_lifetime = int(self.estimate_life_span() * 12)

        # CAPEX:
        energy_dependent_costs = costs_project_development + costs_construction_and_equipment + \
            costs_system_integration + costs_installed_capacity
        power_dependent_costs = costs_grid_integration + costs_controls_and_communication + \
            costs_power_equipment
        initial_costs = energy_dependent_costs * capacity + power_dependent_costs * self.p_market + \
                        property_costs
        installed_costs = costs_power_equipment * self.p_market + costs_installed_capacity * capacity

        # Points of investments:
        investment_costs = [initial_costs]
        points_of_reinvestment = []
        for t in range(1, investment_horizon):
            if(np.mod(t, storage_lifetime) == 0):
                points_of_reinvestment.append(t)

        # NPV of initial and replacement investments:
        npv_investment = initial_costs
        npv_salvage_value = 0
        for i in points_of_reinvestment:
            price_decrease_factor = 1 - (0.025 * i / 12)  # for decreasing battery and PE prices
            npv_investment += ((costs_construction_and_equipment + costs_system_integration + \
                                costs_installed_capacity * price_decrease_factor) * \
                                capacity + costs_power_equipment * price_decrease_factor * self.p_market) * \
                                np.power(1 - capital_costs, i)

            investment_costs.append((costs_construction_and_equipment + costs_system_integration + \
                                     costs_installed_capacity * price_decrease_factor) * \
                                     capacity + costs_power_equipment * price_decrease_factor * self.p_market)
            npv_salvage_value += end_of_life_price * installed_costs * np.power(1 - capital_costs, i)

        if (1 - (investment_horizon - np.max(points_of_reinvestment)) / storage_lifetime > end_of_life_price):
            linear_depreciation_factor = 1 - (investment_horizon - np.max(points_of_reinvestment)) / storage_lifetime
        else:
            linear_depreciation_factor = end_of_life_price

        npv_salvage_value += (linear_depreciation_factor * installed_costs + property_costs) * \
                             np.power(1 - capital_costs, investment_horizon)
        print(npv_salvage_value)
        npv_capex = npv_investment - npv_salvage_value

        # OPEX:
        yearly_expenses_maintenance = []
        yearly_expenses_labour = []
        yearly_expenses_supply_costs = []
        for t in range(investment_horizon):
            # Sauer Lecture OPEX Calculation - Maintenance
            yearly_expenses_maintenance.append(np.power((1 - capital_costs), t) * \
                                               costs_fixed_o_m * capacity * costs_installed_capacity)
            # Labour costs
            yearly_expenses_labour.append(np.power((1 - capital_costs), t) * monitoring_labour_costs)

            npv_opex = np.sum(yearly_expenses_maintenance + yearly_expenses_labour + yearly_expenses_supply_costs)

        # annuity factor
        a = math.pow(1 + capital_costs, investment_horizon)
        annuity_factor = capital_costs * a / (a - 1)

        return {
            'capex': npv_capex * annuity_factor * 12,
            'opex': npv_opex * annuity_factor * 12,
            'total': (npv_capex + npv_opex) * annuity_factor * 12
        }


# round datetime to next x sec
def ceil_dt_sec(dt, n_sec):
    # number of seconds to next quarter hour mark
    delta = math.ceil(dt.second / n_sec) * n_sec - dt.second
    # time + number of seconds to quarter hour mark.
    return dt + timedelta(seconds=delta)

# https://stackoverflow.com/questions/13071384/ceil-a-datetime-to-next-quarter-of-an-hour
# round datetime to next x mins
def ceil_dt_min(dt, n_min):
    # how many secs have passed this hour
    nsecs = dt.minute * 60 + dt.second
    # number of seconds to next quarter hour mark
    delta = math.ceil(nsecs / n_min / 60) * n_min * 60 - nsecs
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


def annualize_and_round(value, year_fraction):
    return round(value / year_fraction, 2)


def get_linear_interpolation(pt_a, pt_b, x):
    m = (pt_a[1] - pt_b[1]) / (pt_a[0] - pt_b[0])
    b = pt_a[1] - m * pt_a[0]

    return m * x + b


def annuitize_yearly_sums(yearly_sums, n_years):
    m = 12
    npv = 0
    years = list(yearly_sums.keys())
    years.sort()
    for i, year in enumerate(years):
        for j, monthly_total in enumerate(yearly_sums[year]):
            npv += monthly_total / math.pow(1 + COST_OF_CAPITAL/m, i * m + j)

    a = math.pow(1 + COST_OF_CAPITAL/m, math.ceil(m*n_years - 1))
    annuity_factor = COST_OF_CAPITAL * a / (a - 1) if a != 1 else 1

    return npv * annuity_factor
