from Battery import Battery
import numpy as np

class StorageSystem:

    DEAD_BAND = 0.01        # allowed control dead band [Hz]
    MAX_DELTA_F = 0.2       # maximum frequency deviation for full power [Hz]
    OVER_FULFILLMENT = 1.2  # maximum over fulfillment allowed [%/100]
    ALERT_DURATION = 0.25   # duration full power must be delivered in the alert state [h]

    def __init__(self, battery=Battery(), p_market=1, capacity=1, p_max=1.25, p_base=1):
        self.battery = battery      # Battery class
        self.p_market = p_market    # marketable power [pu]
        self.capacity = capacity    # battery capacity in [pu * h]
        self.p_max = p_max          # maximum power [pu]
        self.p_base = p_base        # base power [MW]

        # maximum allowed state of charge in normal state [%/100]
        self.soc_max = (capacity - self.ALERT_DURATION * p_market) / capacity

        # minimum allowed state of charge in normal state [%/100]
        self.soc_min = self.ALERT_DURATION * p_market / capacity

    # df : frequency delta in [Hz]
    # dt : time delta [s]
    def execute_step(self, df, dt=1):
        p = self.compute_power(df)

        # energy units % of battery capacity
        e = (p * dt / 3600) / self.capacity

        if e > 0:
            self.battery.discharge(e)
        elif e < 0:
            self.battery.charge(-e)

        self.battery.count_self_discharge_losses(dt)

        return p, e, self.battery.soc

    def compute_power(self, df):
        # TODO active energy management (dead band, over-fulfillment, activation delay)
        return -np.clip(df / self.MAX_DELTA_F, -1, 1)

    def manage_soc(self):
        # TODO active energy management
        pass