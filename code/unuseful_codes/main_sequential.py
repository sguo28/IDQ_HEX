from config.hex_setting import HEX_SHP_PATH, CS_SHP_PATH, NUM_NEAREST_CS, TRIP_FILE, TRAVEL_TIME_FILE, \
    TIMESTEP, START_OFFSET, SIM_DAYS, START_TIME, TRAINING_CYCLE, UPDATE_CYCLE, SAVING_CYCLE
from common.time_utils import get_local_datetime
from simulator.simulator_sequential import Simulator
from dqn_agent.dqn_agent import DeepQNetworkAgent
import numpy as np
import time
# ---------------MAIN FILE---------------

if __name__ == '__main__':

    if SIM_DAYS > 0:
        start_time = START_TIME + int(60 * 60 * 24 * START_OFFSET)  # start_time = 0
        print("Simulate Episode Start Datetime: {}".format(get_local_datetime(start_time)))
        end_time = start_time + int(60 * 60 * 24 * SIM_DAYS)
        print("Simulate Episode End Datetime : {}".format(get_local_datetime(end_time)))

        simulator = Simulator(start_time, TIMESTEP)
        simulator.init(HEX_SHP_PATH, CS_SHP_PATH, TRIP_FILE, TRAVEL_TIME_FILE, NUM_NEAREST_CS)

        dqn_agent = DeepQNetworkAgent()
        n_steps = int(3600 * 24 / TIMESTEP)  # 60 per minute
        with open('logs/parsed_results_21days.csv', 'w') as f, open('logs/training_hist_21days.csv', 'w') as h:
            f.writelines(
                '{},{},{},{},{},{},{},{},{},{},{},{}\n'.format("time", "num_idle", "num_serving", "num_charging",
                                                               "num_cruising", "num_assigned", "num_waitpile",
                                                               "num_tobedisptached", "num_offduty",
                                                               "num_matches", "pass_arrivals", "removed_pass"))
            h.writelines('{},{},{},{}\n'.format("step","loss","reward","len_replay_buffer"))
            for day in range(SIM_DAYS):
                print("############################ SUMMARY ################################")
                for i in range(n_steps):
                    tick = simulator.get_current_time()
                    simulator.par_step_download()

                    vehicle_infos = simulator.get_states()
                    if vehicle_infos is not None:
                        state_batches, num_valid_relos = vehicle_infos
                        simulator.attach_actions_to_vehs(dqn_agent.get_actions(state_batches, num_valid_relos))

                    simulator.par_step_push()
                    num_idle, num_serving, num_charging, num_cruising, num_matches, total_num_arrivals, total_removed_passengers, num_assigned, num_waitpile, num_tobedisptached, num_offduty = simulator.summarize_metrics()

                    f.writelines(
                        '{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(tick, num_idle, num_serving, num_charging,
                                                                       num_cruising, num_assigned, num_waitpile,
                                                                       num_tobedisptached, num_offduty,
                                                                       num_matches, total_num_arrivals,
                                                                       total_removed_passengers,
                                                                       ))
                    # print(
                    #     'Number of matched: {}, number of idling:{}, number of cruising:{}， number of charging:{}'.format(
                    #         num_matches, num_idle, num_cruising, num_charging))

                    states, actions, next_states, rewards = simulator.dump_transition_to_dqn()

                    if states is not None:
                        [dqn_agent.add_transition(state, action, next_state, reward) for
                         state, action, next_state, reward in zip(states, actions, next_states, rewards)]
                        print('average reward is {}'.format(np.mean(rewards)))
                        simulator.all_transitions = []

                    if tick % TRAINING_CYCLE == 0:
                        dqn_agent.train(h)
                    if tick % UPDATE_PARAMETER == 0:
                        dqn_agent.copy_parameter()
                    # for charging_station_id in simulator.get_charging_station_ids():
                    #     f1.writelines('{},{}\n'.format(tick, charging_station_id))
