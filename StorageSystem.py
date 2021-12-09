import math
from collections import deque
from datetime import datetime
from datetime import timedelta

import numpy as np

from Battery import Battery
from Transaction import Transaction


class StorageSystem:
    DEAD_BAND = 0.01  # allowed control dead band [Hz]
    MAX_DELTA_F = 0.2  # maximum frequency deviation for full power [Hz]
    OVER_FULFILLMENT = 1.2  # maximum over fulfillment allowed [%/100]
    ALERT_DURATION = 0.25  # duration full power must be delivered in the alert state [h]

    # Related to primary market transactions
    PRODUCT_DURATION = 0.25  # duration of transaction [h]
    TRANSACTION_DELAY = 0.5  # delay between placing request and delivery [h]
    P_MAX_CONST_DEV = 0.25  # the estimated maximum power required for a sustained deviation [pu] (50 mHz)

    def __init__(self, battery=Battery(), p_market=1, capacity=1, p_max=1.25, p_base=1):
        self.battery = battery  # Battery class
        self.p_market = p_market  # marketable power [pu]
        self.capacity = capacity  # battery capacity in [pu * h]
        self.p_max = p_max  # maximum power [pu]
        self.p_base = p_base  # base power [MW]

        # maximum allowed state of charge in normal state [%/100]
        self.soc_max = (capacity - self.ALERT_DURATION * p_market) / capacity

        # minimum allowed state of charge in normal state [%/100]
        self.soc_min = self.ALERT_DURATION * p_market / capacity

        # SOC when a sell transaction should be initiated to maintain SOC within nominal limits during normal operation
        self.soc_sell_trigger = self.soc_max - (self.TRANSACTION_DELAY * self.P_MAX_CONST_DEV) / self.capacity

        # SOC when a buy transaction should be initiated to maintain SOC within nominal limits during normal operation
        self.soc_buy_trigger = self.soc_min + (self.TRANSACTION_DELAY * self.P_MAX_CONST_DEV) / self.capacity

        self.scheduled_transactions = deque()
        self.transaction_archive = []

    # df : frequency delta in [Hz]
    # t  : current timestamp [datetime]
    # dt : time delta [s]
    def execute_step(self, df, t, dt=1):
        # FCR power (product);
        # Power for SOC management via allowed manipulations (dead band, over-fulfillment, activation delay)
        p_fcr, p_soc_fcr = self.compute_fcr_power(df)
        p_soc_trans = self.manage_soc_trans(t)

        p_batt = p_fcr - p_soc_fcr - p_soc_trans

        # energy units % of battery capacity
        e = (p_batt * dt / 3600) / self.capacity

        if e > 0:
            self.battery.discharge(e)
        elif e < 0:
            self.battery.charge(-e)

        self.battery.count_self_discharge_losses(dt)

        return p_batt, p_fcr, p_soc_fcr, p_soc_trans, self.battery.soc

    def compute_fcr_power(self, df):
        # TODO active energy management (dead band, over-fulfillment, activation delay)
        p_soc = 0
        return -np.clip(df / self.MAX_DELTA_F, -1, 1), p_soc

    def manage_soc_trans(self, t):
        p = 0

        # 1. check for an active transaction (i.e. power delivery)
        # should only be one active transaction at a time!
        for tran in self.scheduled_transactions:
            if t in tran:
                p = tran.power
                break

        # 2. At 15 min intervals,
        #   a. archive old transactions
        #   b. check if a new transaction is needed!
        if t == ceil_dt(t):
            self.archive_transactions(t)
            self.create_transaction(t)

        return p

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

    def create_transaction(self, t):
        # find energy of currently schedule transactions
        energy = 0
        for tran in self.scheduled_transactions:
            remaining_hours = min(tran.end_time - t, tran.end_time - tran.start_time) / timedelta(hours=1)
            energy += remaining_hours * tran.power
            if remaining_hours < 0:
                raise "Transaction not properly archived!"

        delta_soc = energy / self.capacity

        if self.battery.soc + delta_soc > self.soc_sell_trigger:
            self.scheduled_transactions.append(Transaction(
                start_time=t+timedelta(hours=self.TRANSACTION_DELAY),
                duration=timedelta(hours=self.PRODUCT_DURATION),
                power=-self.P_MAX_CONST_DEV
            ))
        elif self.battery.soc + delta_soc < self.soc_buy_trigger:
            self.scheduled_transactions.append(Transaction(
                start_time=t+timedelta(hours=self.TRANSACTION_DELAY),
                duration=timedelta(hours=self.PRODUCT_DURATION),
                power=self.P_MAX_CONST_DEV
            ))


# https://stackoverflow.com/questions/13071384/ceil-a-datetime-to-next-quarter-of-an-hour
# round datetime to next 15 mins
def ceil_dt(dt):
    # how many secs have passed this hour
    nsecs = dt.minute * 60 + dt.second
    # number of seconds to next quarter hour mark
    delta = math.ceil(nsecs / 900) * 900 - nsecs
    # time + number of seconds to quarter hour mark.
    return dt + timedelta(seconds=delta)
