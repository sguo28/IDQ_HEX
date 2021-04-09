import time
import numpy as np
import pandas as pd
from common.time_utils import get_local_datetime
from config.hex_setting import HEX_SHP_PATH, CS_SHP_PATH, NUM_NEAREST_CS, TRIP_FILE, TRAVEL_TIME_FILE, \
    TIMESTEP, START_OFFSET, SIM_DAYS, START_TIME, TRAINING_CYCLE, UPDATE_CYCLE, STORE_TRANSITION_CYCLE, CNN_RESUME
from dqn_agent.dqn_agent_with_cnn.cnn_dqn_agent import DeepQNetworkAgent
from simulator.simulator_cnn import Simulator
from logs.parse_results_cnn import auto_save_metric_plots
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

        with open('logs/parsed_results_cnn_local.csv', 'w') as f, open('logs/target_charging_stations_cnn_local.csv', 'w') as g, open('logs/training_hist_cnn_local.csv', 'w') as h, open('logs/demand_supply_gap_cnn_local.csv', 'w') as l1, open('logs/cruising_od_cnn_local.csv', 'w') as m1, open('logs/matching_od_cnn_local.csv', 'w') as n1:
            # if not CNN_RESUME: # if not continuing training: set up column names:
            f.writelines(
                '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format("time", "num_idle", "num_serving",
                                                                        "num_charging",
                                                                        "num_cruising", "num_assigned",
                                                                        "num_waitpile",
                                                                        "num_tobedisptached", "num_offduty",
                                                                        "num_matches", "pass_arrivals",
                                                                        "longwait_pass",
                                                                        "served_pass", "removed_pass",
                                                                        "consumed_SOC_per_cycle",
                                                                        "average_cumulated_earning"))
            g.writelines('{},{},{}\n'.format("tick", "cs_id","destination_cs_id"))
            h.writelines('{},{},{},{},{},{}\n'.format("step", "loss", "reward", "learning_rate","sample_reward","sample_SOC"))
            l1.writelines('{},{},{}\n'.format("step", "hex_zone_id", "demand_supply_gap"))
            m1.writelines('{},{},{}\n'.format("step","origin_hex","destination_hex"))
            n1.writelines('{},{},{}\n'.format("step", "origin_hex", "destination_hex"))

            for day in range(SIM_DAYS):
                print("############################ DAY {} SUMMARY ################################".format(day))
                for i in range(n_steps):
                    tick = simulator.get_current_time()
                    start_tick = time.time()
                    simulator.step()
                    t1 = time.time() - start_tick
                    local_state_batches, num_valid_relos = simulator.get_local_states()
                    # t2 = time.time() - start_tick
                    global_state = simulator.get_global_state()
                    # t3 = time.time() - start_tick
                    # if tick >0 and np.sum(global_state) == 0: # check if just reset
                    #     global_state = global_state_slice
                    if len(local_state_batches) > 0:
                        simulator.attach_actions_to_vehs(
                            dqn_agent.get_actions(local_state_batches, num_valid_relos, global_state))
                    # t4 = time.time() - start_tick
                    simulator.update()  # update time, get metrics.
                    # t5 = time.time() - start_tick
                    (num_idle, num_serving, num_charging, num_cruising, n_matches, total_num_arrivals,
                     total_removed_passengers, num_assigned, num_waitpile, num_tobedisptached, num_offduty,
                     average_reduced_SOC, total_num_longwait_pass, total_num_served_pass, average_cumulated_earning) = simulator.summarize_metrics(l1, g, m1, n1)

                    f.writelines(
                        '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(tick, num_idle, num_serving,
                                                                                num_charging,
                                                                                num_cruising, num_assigned,
                                                                                num_waitpile,
                                                                                num_tobedisptached, num_offduty,
                                                                                n_matches,
                                                                                total_num_arrivals,
                                                                                total_num_longwait_pass,
                                                                                total_num_served_pass,
                                                                                total_removed_passengers,
                                                                                average_reduced_SOC,
                                                                                average_cumulated_earning))
                    # dump transitions to DQN module
                    if tick % STORE_TRANSITION_CYCLE == 0:
                        simulator.store_transitions_from_veh()
                        states, actions, next_states, rewards, time_steps, valid_action_nums_ = simulator.dump_transition_to_dqn()
                        if states is not None:
                            [dqn_agent.add_transition(states, actions, next_states, rewards, time_steps, valid_action_num_) for
                             states, actions, next_states, rewards, time_steps, valid_action_num_ in zip(states, actions, next_states, rewards, time_steps, valid_action_nums_)]
                            print('For tick {}, average reward is {}'.format(tick/60,np.mean(rewards)))
                        dqn_agent.add_global_state_dict(simulator.global_state_tensor)  # a 4-dim np array

                        # now reset transition and global state
                        simulator.reset_storage()
                    t6 = time.time() - start_tick

                    t_start = time.time()
                    if tick % TRAINING_CYCLE == 0:
                        dqn_agent.train(h)

                    if tick % UPDATE_CYCLE == 0:
                        dqn_agent.copy_parameter()
                    t7 = time.time() - t_start
                    print('Iteration {} completed, duration: {:.3f} and training: {:.3f}'.format(tick / 60, t1,t7))
                    # print('Iteration {} completed, Durations: store={:.3f}; step={:.3f}; get states ={:.3f}; attach state = {:.3f}; update time = {:.3f}; dump to replaybuffer = {:.3f}; training = {:.3f}'.format(tick / 60, t0,t1,t3,t4,t5, t6,t7))

                    if tick % (60*60*24*5) == 0 and tick != 0: # plot per 5 days while rolling over on per day.
                        metric_df = pd.read_csv('logs/parsed_results_cnn_local.csv')
                        train_df = pd.read_csv('logs/training_hist_cnn_local.csv')
                        auto_save_metric_plots(metric_df, train_df, tick //(60*60*24*5))
