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
DT = 5          # seconds

# Decimation factor for logging economic sim
ECON_LOG_DEC = 12

# level to inc/red transaction triggers when adjusting them in order to meet PQ requirements
TRAN_TRIGGER_ADJUST = 0.02


# optimize configurations
def optimize_configurations():
    batteries = [
        Battery.LFPBattery
    ]

    systems = []
    capacities = []
    for battery_type in batteries:
        systems, capacities = create_configurations(battery_type=battery_type)

    # get non-duplicated list
    capacities = set(capacities)

    duration = None
    if DEV:
        duration = timedelta(days=2)

    valid_systems = validate_configurations(systems, duration=duration)

    results = run_economic_simulation(valid_systems, duration=duration)
    results.sort()

    Plotter.plot_summary(results, capacities, save_fig=True, path='figures/')


def create_configurations(soc_target=0.5, battery_type=None, battery_args=None):
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
        plot_path='figures/economic/'
    )

    return results


def run_parallel_simulations(data_src, duration, dt, storage_systems, plot_path=None):
    shm_links, freq = load_sharable_block(data_src=data_src, duration=duration, dt=dt)

    systems = map(lambda ss: (freq, dt, ss, plot_path), storage_systems)

    results = []
    try:
        with Pool(processes=min(8, cpu_count())) as pool:
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


def simulate_storage_config(freq_info, dt, ss, plot_path=None):

    index_shm = shared_memory.SharedMemory(name=freq_info['index'])
    freq_shm = shared_memory.SharedMemory(name=freq_info['freq'])
    index = np.ndarray((freq_info['size'],), dtype='datetime64[s]', buffer=index_shm.buf)
    freq = np.ndarray((freq_info['size'],), dtype=np.float32, buffer=freq_shm.buf)

    ss.init_sim_data(freq.size)
    current_day = None
    for i in range(0, freq.size):
        t1 = pd.Timestamp(index[i])
        ss.execute_step(freq=freq[i], dt=dt, t=t1)
        if current_day is None or current_day != t1.date().day:
            current_day = t1.date().day
            # print("Currently evaluating {}".format(t1.date()))

    if plot_path is not None:
        Plotter.plot_time_curves(ss, save_fig=True, path=plot_path)
        Plotter.plot_rel_freq_data(ss, save_fig=True, path=plot_path)

    index_shm.close()
    freq_shm.close()

    return ss
