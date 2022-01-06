import math

import numpy as np

# Update the battery capacity due to aging after n seconds
UPDATE_CAPACITY_EVERY_N_S = 3600  # every hour


class Battery:

    # maximum expected battery lifespan in years
    CALENDER_LIFESPAN = 12

    # The battery is modeled at a high level with considerations for:
    #   (i)     Energy
    #   (ii)    Power
    #   (iii)   Efficiency
    #
    # Assumptions: constant voltage, negligible impact of SOC on ageing
    #
    # eta_char [%/100]
    # eta_disc [%/100]
    # eta_self_disc [%/100 per day]
    def __init__(self, soc=0.5, eta_char=1, eta_disc=1, eta_self_disc=0, capacity_nominal=1):
        self.name = "Generic"
        self.starting_soc = soc                                 # starting state of charge [%/100]
        self.soc = soc                                          # state of charge [%/100]
        self.eta_char = eta_char                                # charging efficiency [%/100]
        self.eta_disc = eta_disc                                # discharging efficiency [%/100]
        self.eta_self_disc_s = eta_self_disc / (3600 * 24)      # self discharge [%/100 per s]
        self.eq_full_cycle_count = 0                            # equivalent full cycle count
        self.capacity_nominal = capacity_nominal                # nominal capacity [MW]
        self.capacity_actual = capacity_nominal                 # actual usable capacity [MW]

        # For cycle counting
        self.cycles = {
            'cycle_depth': [],
            'c_rate': []
        }
        self.current_cycle = {
            'soc': [],
            'c_rate': []
        }
        self.accumulated_losses = 0
        self.elapsed_time = 0

    # compare batteries to each other for equivalent parameters (floats only)
    def __eq__(self, other):
        if not isinstance(other, Battery):
            raise NotImplementedError

        # examine only float parameters
        for attr in self.__dict__.keys():
            value = getattr(self, attr)
            if type(value) in [float, np.float64]:
                if getattr(other, attr) != value:
                    return False
        return True

    def reset(self):
        self.soc = self.starting_soc
        self.cycles['cycle_depth'] = []
        self.cycles['c_rate'] = []
        self.current_cycle['soc'] = []
        self.current_cycle['c_rate'] = []
        self.accumulated_losses = 0
        self.eq_full_cycle_count = 0
        self.capacity_actual = self.capacity_nominal
        self.elapsed_time = 0

    #   energy [MWh]
    #   power [MWh]
    #   dt [s]
    def execute_step(self, energy, power, dt):
        net_energy = 0
        net_power = 0

        # Charging
        if energy < 0:
            net_energy = energy * self.eta_char
            net_power = power * self.eta_char

        # Discharging
        elif energy > 0:
            net_energy = energy / self.eta_disc
            net_power = power / self.eta_disc

        pu_energy = net_energy / self.capacity_actual

        # self discharge
        # neglect power, as self discharge is internal
        pu_energy += dt * self.eta_self_disc_s

        self.soc -= pu_energy
        self.track_cycles(-pu_energy, net_power / self.capacity_actual)
        self.accumulated_losses -= energy - pu_energy * self.capacity_actual   # positive value

        if np.isnan(self.accumulated_losses):
            raise Exception('loss computation error')

        if self.elapsed_time % UPDATE_CAPACITY_EVERY_N_S < dt:
            self.update_capacity()
        self.elapsed_time += dt

        return net_energy, net_power

    def track_cycles(self, pu_energy, c_rate):
        # Basic equivalent full cycle method:
        # Comparison of Lead-Acid and Li-Ion Batteries Lifetime
        # Prediction Models in Stand-Alone Photovoltaic Systems
        # under simplified model
        self.eq_full_cycle_count += np.abs(pu_energy) / 2

        # Approach based on:
        # Fundamentals of Using Battery Energy Storage Systems to
        # Provide Primary Control Reserves in Germany, section 5.7.2

        # (1a) Change from idle mode to charge/discharge;
        if c_rate != 0 and len(self.current_cycle['soc']) == 0:
            self.current_cycle['soc'].append(self.soc)
            self.current_cycle['c_rate'].append(c_rate)

        # (1b) Change to idle mode
        elif c_rate == 0 and len(self.current_cycle['soc']) > 0:
            self.add_half_cycle()

        # (1c) idle mode
        elif c_rate == 0:
            pass

        # (2) Sign change of load (from charge to discharge or vice versa);
        elif np.sign(c_rate) != np.sign(self.current_cycle['c_rate'][0]):
            self.add_half_cycle()
            self.current_cycle['soc'].append(self.soc)
            self.current_cycle['c_rate'].append(c_rate)

        # # (3) Relatively strong change of gradient during charge or discharge (i.e., DC rate = 0.6).
        # elif np.abs(c_rate - self.current_cycle['c_rate'][0]) > 0.6:
        #     self.add_half_cycle()
        #     self.current_cycle['soc'].append(self.soc)
        #     self.current_cycle['c_rate'].append(c_rate)

        else:
            self.current_cycle['soc'].append(self.soc)
            self.current_cycle['c_rate'].append(c_rate)

    def add_half_cycle(self):
        cycle_depth = self.soc - self.current_cycle['soc'][0]
        self.cycles['cycle_depth'].append(cycle_depth)
        self.cycles['c_rate'].append(
            max(min(self.current_cycle['c_rate']), max(self.current_cycle['c_rate']), key=abs)
        )
        self.current_cycle['soc'].clear()
        self.current_cycle['c_rate'].clear()

    def update_capacity(self):
        pass    # implement in child class

    def estimate_lifespan(self, time_span_years):
        pass    # implement in child class


class LFPBattery(Battery):

    CALENDER_LIFESPAN = 12

    def __init__(self, **kwargs):
        kwargs['eta_char'] = 0.90
        kwargs['eta_disc'] = 0.90
        kwargs['eta_self_disc'] = 0
        super().__init__(**kwargs)
        self.name = "LFP"   # must be unique

    def estimate_lifespan(self, year_fraction):
        # calendar lifetime:
        soc = 50    # SOC in %
        T = 30      # Temperature in C°
        N = 20      # aging time frame
        t = N*12

        # fitted curve for calendar aging
        calendar_capacity_fade = np.zeros(t)
        for t in range(t):
            calendar_capacity_fade[t] = 0.0025 * pow(np.e, 0.1099 * T) * pow(np.e, 0.0169 * soc) * \
                                        pow(t, (-3.866 * pow(10, -13)) * pow(T, 6.635) +
                                            (-4.853 * pow(10, -12)) * pow(soc, 5.508) + 0.9595) + 0.7

        for i, c in enumerate(calendar_capacity_fade):
            if (c >= 20):
                self.CALENDER_LIFESPAN = i
                break

        # cycle lifetime
        doc = 0.8   # cycle depth of used data for fitted curve
        c_rate = 0.2    # conservative c_rate due to too low c_rates for fitted curve
        fec = self.eq_full_cycle_count / year_fraction  # cycles per year

        k_crate = 0.0630 * c_rate + 0.0971  # c-rate dependent factor
        k_doc = 4.02 * pow(doc - 0.6, 3) + 1.0923   # dod dependent factor
        k_T = 1     # temperature dependent factor(1 for ambient temperatures)

        cycle_capacity_fade = np.zeros(t)
        for i, t in enumerate(range(t)):
            k_fec = pow(fec * t/12, 0.5)    # full equivalent cycle dependent factor
            cycle_capacity_fade[i] = k_crate * k_doc * k_fec * k_T

        combined_lifetime = 0
        for i in range(t):
            comb_fade = calendar_capacity_fade[i] + cycle_capacity_fade[i]
            if (comb_fade >= 20):
                combined_lifetime = i
                break

        return(combined_lifetime/12)
        #return min(self.CALENDER_LIFESPAN, 3000 / self.eq_full_cycle_count * year_fraction)

    def update_capacity(self):
        if self.elapsed_time == 0:
            return  # no degradation

        t = self.elapsed_time / (3600 * 24 * 365)
        temp = 30      # Celsius
        soc = 0.5      # todo
        cap_fade_cal = 0.0025 * (math.e ** (0.1099 * temp)) * (math.e ** (0.0169 * soc)) \
                       * t ** (-3.866e-13 * temp ** 6.635 - 4.853e-12 * soc ** 5.508 + 0.9595) + 0.7

        # cap_fade_cycle = 20 * self.eq_full_cycle_count / 3000
        # cycle lifetime
        doc = 0.8  # cycle depth of used data for fitted curve
        c_rate = 0.2  # conservative c_rate due to too low c_rates for fitted curve

        k_crate = 0.0630 * c_rate + 0.0971  # c-rate dependent factor
        k_doc = 4.02 * pow(doc - 0.6, 3) + 1.0923  # dod dependent factor
        k_temp = 1  # temperature dependent factor(1 for ambient temperatures)
        k_fec = pow(self.eq_full_cycle_count, 0.5)

        cap_fade_cycle = k_crate * k_doc * k_fec * k_temp

        cap_fade = (cap_fade_cycle + cap_fade_cal) / 100

        self.capacity_actual = self.capacity_nominal * (1 - cap_fade)

