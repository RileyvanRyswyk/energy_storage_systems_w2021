import glob
import json
import math
from datetime import timedelta
from multiprocessing import cpu_count, Pool, shared_memory

import numpy as np
import pandas as pd
import tqdm

import Battery
from FrequencyData import FrequencyData
from MarketData import MarketData
import StorageSystem
import Plotter

DEV = False     # limit data for faster processing
DT = 30         # seconds

# Decimation factor for logging economic sim
ECON_LOG_DEC = 5 * 60 / DT

# level to inc/red transaction triggers when adjusting them in order to meet PQ requirements
TRAN_TRIGGER_ADJUST = 0.01

# where valid configurations are stored
VALID_CONFIGS_FILE = 'data/valid_configs/valid_configs.json'


# optimize configurations
def optimize_configurations():
    valid_systems = get_valid_configurations()

    duration = None
    if DEV:
        duration = timedelta(days=2)

    results = run_economic_simulation(valid_systems, duration=duration)
    results.sort()

    Plotter.plot_summary(results, save_fig=True, path='figures/')


def run_economic_simulation(systems, duration=None):
    print("Evaluating economics for {} configurations.".format(len(systems)))

    # market data is too small to use shared memory
    md = MarketData()
    for system in systems:
        system.link_market_data(md)
        system.log_decimation = ECON_LOG_DEC

    results = run_parallel_simulations(
        data_src=FrequencyData.DATA_PATH + '/202*_Frequenz.csv',
        duration=duration,
        dt=DT,
        storage_systems=systems,
        plot_path='figures/economic/',
        till_death=True
    )

    return results


def get_valid_configurations():
    files = glob.glob("{}".format(VALID_CONFIGS_FILE))

    if len(files) != 1:
        create_configurations()

    valid_systems = []
    with open(VALID_CONFIGS_FILE) as f:
        system_configs = json.load(f)
        import Battery
        for battery_type, system_config in system_configs.items():
            for size, params in system_config['sizes'].items():
                battery_class_name = getattr(Battery, battery_type)
                battery = battery_class_name(capacity_nominal=float(size))
                ss = StorageSystem.StorageSystem(
                    battery=battery,
                    p_market=system_config['p_market'],
                    p_max=system_config['p_max'],
                    soc_target=system_config['soc_target'],
                    validation_data=system_config['sizes']
                )
                valid_systems.append(ss)

    return valid_systems


def create_configurations():
    batteries = [
        Battery.LFPBattery
    ]

    systems = []
    for battery_type in batteries:
        systems, capacities = init_unvalidated_configurations(battery_type=battery_type)

    duration = None
    if DEV:
        duration = timedelta(days=2)

    valid_systems = validate_configurations(systems, duration=duration)
    valid_systems.sort()
    store_valid_systems(valid_systems)


def init_unvalidated_configurations(soc_target=0.5, battery_type=None, battery_args=None):
    if battery_type is None:
        battery_type = Battery.Battery
    if battery_args is None:
        battery_args = {}

    p_market = 5
    p_max = 1.25 * p_market

    # compute the minimum storage size in 0.25 MWh increments
    weighted_duration = StorageSystem.ALERT_DURATION
    weighted_duration += StorageSystem.TRANSACTION_DELAY * StorageSystem.P_MAX_CONST_DEV
    e_min = 2 * p_market * weighted_duration
    e_min = math.ceil(e_min * 4) / 4.0  # 0.25 MWh increments

    capacities = np.arange(e_min, 1.5*e_min, 0.25)
    systems = []
    for capacity in capacities:
        kwargs = battery_args.copy()
        kwargs['capacity_nominal'] = capacity
        battery = battery_type(**kwargs)
        ss = StorageSystem.StorageSystem(
            battery=battery,
            p_market=p_market,
            p_max=p_max,
            soc_target=soc_target,
        )
        systems.append(ss)

    return systems, capacities


#  Run configurations against the prequalification (PQ) data
#  and verify that the min or max state of charge was never reached,
#  since the system never enters the alert state with this dataset
def validate_configurations(systems, depth=1, duration=None):
    print("Validating {} capacities (depth={}).".format(len(systems), depth))

    results = run_parallel_simulations(
        data_src=FrequencyData.PQ_DATA,
        duration=duration,
        dt=DT,
        storage_systems=systems,
        plot_path='figures/validation/'
    )

    valid_systems = []
    systems_to_redo = []
    for index, result in enumerate(results):
        if np.min(result.sim_data['batt_soc']) > result.soc_min \
         and np.max(result.sim_data['batt_soc']) < result.soc_max:
            result.reset()
            valid_systems.append(result)
        elif result.soc_sell_trigger > 0.5 + TRAN_TRIGGER_ADJUST:
            result.reset()
            result.soc_sell_trigger -= TRAN_TRIGGER_ADJUST
            result.soc_buy_trigger += TRAN_TRIGGER_ADJUST
            systems_to_redo.append(result)

    print("{} systems are valid (depth={}).".format(len(valid_systems), depth))
    if len(systems_to_redo) > 0:
        return valid_systems + validate_configurations(systems_to_redo, depth=depth+1)
    else:
        return valid_systems


# Store valid configurations for later use in the economic portion
def store_valid_systems(valid_systems):
    systems = {}
    for system in valid_systems:
        battery = type(system.battery).__name__
        if battery not in systems:
            systems[battery] = {
                'p_market': system.p_market,
                'p_max': system.p_max,
                'soc_target': system.soc_target,
                'sizes': {}
            }
        systems[battery]['sizes'][system.battery.capacity_nominal] = {
            'soc_max': system.soc_max,
            'soc_min': system.soc_min,
            'buy': system.soc_buy_trigger,
            'sell': system.soc_sell_trigger
        }

    with open(VALID_CONFIGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(systems, f, ensure_ascii=False, indent=4)


def run_parallel_simulations(data_src, duration, dt, storage_systems, plot_path=None, till_death=False):
    shm_links, freq = load_sharable_block(data_src=data_src, duration=duration, dt=dt)

    systems = map(lambda ss: (freq, dt, ss, plot_path, till_death), storage_systems)

    results = []
    try:
        with Pool(processes=min(9, cpu_count())) as pool:
            for result in tqdm.tqdm(
                pool.imap_unordered(simulate_storage_config_star, systems),
                total=len(storage_systems)
            ):
                results.append(result)

            # res = pool.starmap_async(simulate_storage_config, systems, callback=log_result)
            # print("Final result: ", res.get())
    except Exception as err:
        for shm_link in shm_links:
            shm_link.close()
            shm_link.unlink()

        raise f"Unexpected {err=}, {type(err)=}"

    return results


def log_result(result):
    print("Succesfully get callback! With result: ", result)


def load_sharable_block(data_src, duration, dt):
    fd = FrequencyData(data_src)
    data = fd.get_data_subset(duration=duration, dt=dt)
    index = data.index.to_numpy()
    freq = data['freq'].to_numpy()

    index_shm, index_arr = create_shared_block(index, 'datetime64[s]')
    freq_shm, _ = create_shared_block(freq, np.float32)

    freq_info = {'size': index_arr.size, 'index': index_shm.name, 'freq': freq_shm.name}
    shm_links = [index_shm, freq_shm]

    # clean up memory
    del index, freq, data, fd

    return shm_links, freq_info


def create_shared_block(src_array, dtype):
    shm = shared_memory.SharedMemory(create=True, size=src_array.nbytes)
    # Now create a NumPy array backed by shared memory
    np_array = np.ndarray(src_array.shape, dtype=dtype, buffer=shm.buf)
    np_array[:] = src_array[:]  # Copy the original data into shared memory
    return shm, np_array


def simulate_storage_config_star(args):
    return simulate_storage_config(*args)


def simulate_storage_config(freq_info, dt, ss, plot_path=None, till_death=False):

    index_shm = shared_memory.SharedMemory(name=freq_info['index'])
    freq_shm = shared_memory.SharedMemory(name=freq_info['freq'])
    index = np.ndarray((freq_info['size'],), dtype='datetime64[s]', buffer=index_shm.buf)
    freq = np.ndarray((freq_info['size'],), dtype=np.float32, buffer=freq_shm.buf)

    end_of_life = False
    n = 0
    time_increment = pd.Timestamp(index[-1]) - pd.Timestamp(index[0]) + timedelta(seconds=dt)
    current_month = pd.Timestamp(index[0]).month
    try:
        while not end_of_life:
            ss.init_sim_data(freq.size, extend=True)
            for i in range(0, freq.size):
                t1 = pd.Timestamp(index[i])
                t2 = t1 + n * time_increment
                ss.execute_step(freq=freq[i], dt=dt, t=t1, t_star=t2)

                if t1.month != current_month and till_death:
                    current_month = t1.month
                    end_of_life = ss.compute_soc_limits()
                    if end_of_life:
                        ss.trim_sim_data()
                        raise StopIteration

            if till_death:
                n += 1
                if DEV and n > 2:
                    raise StopIteration
            else:
                raise StopIteration
    except StopIteration:
        pass

    if plot_path is not None:
        Plotter.plot_time_curves(ss, save_fig=True, path=plot_path)
        Plotter.plot_rel_freq_data(ss, save_fig=True, path=plot_path)

    index_shm.close()
    freq_shm.close()

    return ss
