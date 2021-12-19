import numpy as np

class Battery:

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
        self.starting_soc = soc                                 # starting state of charge [%/100]
        self.soc = soc                                          # state of charge [%/100]
        self.eta_char = eta_char                                # charging efficiency [%/100]
        self.eta_disc = eta_disc                                # discharging efficiency [%/100]
        self.eta_self_disc_s = eta_self_disc / (3600 * 24)      # self discharge [%/100 per s]
        self.eq_full_cycle_count = 0                            # equivalent full cycle count
        self.capacity_nominal = capacity_nominal                # nominal capacity [MW]

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
            net_power = power / self.eta_char

        pu_energy = net_energy / self.capacity_nominal

        # self discharge
        # neglect power, as self discharge is internal
        pu_energy += dt * self.eta_self_disc_s

        self.soc -= pu_energy
        self.track_cycles(-pu_energy, power / self.capacity_nominal)
        self.accumulated_losses -= energy - pu_energy * self.capacity_nominal   # positive value

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

