from config.hex_setting import hex_shp_path,charging_station_shp_path,num_reachable_hex,NUM_NEAREST_CS, trip_file, travel_time_file, MAP_HEIGHT, MAP_WIDTH
import numpy as np
from common.time_utils import get_local_datetime
from common import mesh
from novelties import status_codes
from logger import sim_logger
from simulator.settings import FLAGS
from dummy_agent.demand_loader import DemandLoader
from dummy_agent.matching_policy import GreedyMatchingPolicy
from dummy_agent.dispatch_policy import DispatchPolicy
from simulator.models.charging_pile.charging_pile_repository import ChargingRepository
from dummy_agent.agent import Dummy_Agent, DQN_Agent
from central_agent import central_agent
from config.hex_setting import TIMESTEP, ENTERING_TIME_BUFFER
import simulator.simulator as sim
#---------------MAIN FILE---------------

class simulator_driver(object):
    '''
    INPUT: 
            training period, training step, matching policy, (learned) dispatching policy, pricing policy 
    AGENTS: 
            dummy agents don't actually dispatch
            dqn agents learn the dispatching policy
    INSTALL:
            conda install skimage
            conda install -c anaconda sqlalchemy
            conda install -c conda-forge polyline
    '''
    # For DQN
    def __init__(self, start_time, timestep, matching_policy, dispatch_policy):
        self.simulator = sim.Simulator(start_time, timestep)
        # For DQN
        self.dqn_agent = DQN_Agent(dispatch_policy)
        self.dummy_agent = Dummy_Agent(dispatch_policy=None)
        self.central_agent = central_agent.Central_Agent(matching_policy)
        self.charging_stations = ChargingRepository()
        self.last_vehicle_id = 1
        self.vehicle_queue = []

    def sample_initial_locations(self, t):
        locations = [mesh.convert_xy_to_lonlat(x, y)[::-1] for x in range(MAP_WIDTH) for y in range(MAP_HEIGHT)]
        p = DemandLoader.load_demand_profile(t) # a matrix of M x N
        p = p.flatten() / p.sum()
        vehicle_locations = [locations[i] for i in np.random.choice(len(locations), size=FLAGS.vehicles, p=p)]
        return vehicle_locations

    def populate_vehicles(self, vehicle_locations):
        n_vehicles = len(vehicle_locations)
        vehicle_ids = range(self.last_vehicle_id, self.last_vehicle_id + n_vehicles)
        self.last_vehicle_id += n_vehicles

        t = self.simulator.get_current_time()
        entering_time = np.random.uniform(t, t + ENTERING_TIME_BUFFER, n_vehicles).tolist()
        q = sorted(zip(entering_time, vehicle_ids, vehicle_locations))
        self.vehicle_queue = q

    def enter_market(self):
        t = self.simulator.get_current_time()
        while self.vehicle_queue:
            t_enter, vehicle_id, location = self.vehicle_queue[0]
            if t >= t_enter:
                self.vehicle_queue.pop(0) # no longer queueing
                self.simulator.populate_vehicle(vehicle_id, location)
            else:
                break

    def get_charging_stations(self):
        return self.charging_stations.get_all()

    def get_current_time(self):
        return self.simulator.get_current_time()

if __name__ == '__main__':
    if FLAGS.days > 0:
        start_time = FLAGS.start_time + int(60 * 60 * 24 * FLAGS.start_offset) # non-zero start-offset means to start after warming.
        # print(start_time, MAX_DISPATCH_CYCLE, MIN_DISPATCH_CYCLE)
        print("Simulate Episode Start Datetime: {}".format(get_local_datetime(start_time)))
        end_time = start_time + int(60 * 60 * 24 * FLAGS.days)
        print("Simulate Episode End Datetime : {}".format(get_local_datetime(end_time)))

        # For DQN
        Sim_experiment = simulator_driver(start_time, TIMESTEP, GreedyMatchingPolicy(), dispatch_policy = DispatchPolicy())
        Sim_experiment.simulator.init_zones(hex_shp_path,charging_station_shp_path,trip_file, travel_time_file,NUM_NEAREST_CS)
        n_steps = int(3600 * 24 / TIMESTEP) # 60 per minute
        buffer_steps = int(3600 / TIMESTEP) # buffer for fleet's enter time
        
        vehicle_locations = Sim_experiment.sample_initial_locations(Sim_experiment.simulator.get_current_time() + 3600 * 3)
                
        Sim_experiment.populate_vehicles(vehicle_locations)
        for day in range(FLAGS.days):
            # with open('logs/output_charge_day%d.csv'%day,'w') as f:
            #     f.writelines('{},{},{},{},{},{},{}\n'.format("time","cs_id","lat","lon","wait_time","queue_len","served_num"))
            sum_avg_cust = 0
            # sum_avg_profit = 0
            sum_avg_wait = 0
            sum_requests = 0
            sum_accepts = 0
            sum_rejects = 0
            prev_rejected_req = []
            print("############################ SUMMARY ################################")
            for i in range(n_steps): # every tick
                # time1=time.time()
                Sim_experiment.enter_market()       #;time2 = time.time()
                print('enter finish')
                Sim_experiment.simulator.step()     #;time3 = time.time()
                print('Step now, current time tick={}'.format(i))
                    
                ''' 
                vehicles = Sim_experiment.simulator.get_vehicles_state()    #;time4= time.time()

                requests = Sim_experiment.simulator.get_new_requests()      #;time5= time.time(); 
                # print('enter time cost',round(time2-time1,3),'s',"#############","simulator_step time cost",round(time3-time2,3), "s","##############","get_new_requests() time cost", round(time5-time4,3),"s")

                charging_stations = Sim_experiment.simulator.get_charging_stations()
                col_names = requests.columns.values

                sum_requests += len(requests)
                requests = requests.set_index("id")
                # print("R After: ", len(requests))

                current_time = Sim_experiment.simulator.get_current_time()

                # For DQN
                if len(vehicles) == 0:
                    continue
                else:
                                        
                    # time6 = time.time()
                    m_commands,c_commands, vehicles = Sim_experiment.central_agent.get_match_commands(current_time, vehicles, requests, charging_stations)

                    dqn_v = vehicles[vehicles.agent_type == agent_codes.dqn_agent]
                    dummy_v = vehicles[vehicles.agent_type == agent_codes.dummy_agent]
                    # print("DQN: ", len(dqn_v), " Dummy: ", len(dummy_v)) ; time7 = time.time()

                    # For DQN and Dummy: these are unmatched and SOC>20% agents after matching/inspecting SOC 
                    d1_commands = Sim_experiment.dummy_agent.get_dispatch_commands(current_time, dummy_v)
                    d2_commands = Sim_experiment.dqn_agent.get_dispatch_commands(current_time, dqn_v)
                    # print("Dummy agent: ", len(d1_commands), " DQN agent: ", len(d2_commands))
                    # time8 = time.time();print("MATCH time",round(time7-time6,3),"s","#####","DISPATCH time",round(time8-time7,3),"s")
                    
                    #all dispatching commands
                    all_commands = d1_commands + d2_commands
                    # print("A: ", len(all_commands), " 1: ", len(d1_commands), " 2: ", len(d2_commands))
                    
                    # changed to assigned
                    prev_rejected_req, accepted_commands, charging_commands = Sim_experiment.simulator.match_vehicles(m_commands,c_commands, Sim_experiment.dqn_agent, Sim_experiment.dummy_agent)

                    # record charging_commands############
                    # For DQN
                    # changed to crusing
                    Sim_experiment.simulator.dispatch_vehicles(all_commands)

                    for cid,cp in enumerate(charging_stations):
                        f.writelines('{},{},{},{},{},{},{}\n'.format((current_time-start_time//60),cid,cp.get_cs_location()[0],cp.get_cs_location()[1],cp.get_average_waiting_time(),cp.get_queue_length(),cp.get_served_num()))
                        if cid == 531:
                            print((current_time-start_time)//60,cid,cp.get_average_waiting_time(),cp.get_queue_length(),cp.get_served_num())                
                    if len(vehicles)>FLAGS.vehicles:
                        print("WARNING ADDITIONAL AGENTS",current_time,len(vehicles))
                    if (len(m_commands) == 0):
                        print("ERR!", len(vehicles), len(requests))

                    avg_cap = 0
                    capacity = []

                    for index, v in vehicles.iterrows():
                        if v.status == status_codes.V_OCCUPIED:
                            capacity.append(v.current_capacity)

                    if len(capacity):
                        avg_cap = np.sum(capacity) / len(capacity)
                        sum_avg_cust += avg_cap
                        # print(len(capacity), sum_avg_cust)

                    net_v = vehicles[vehicles.status != status_codes.V_OFF_DUTY]
                    occ_v = net_v[net_v.status == status_codes.V_OCCUPIED]

                    if len(occ_v) != len(capacity):
                        print("Watch Occupied", len(occ_v), len(capacity))

                    
                    if len(m_commands) > 0:
                        average_wt = np.mean([command['duration'] for command in m_commands]).astype(int)
                    else:
                        average_wt = 0

                    sum_avg_wait += average_wt # accumulate every tick

                    # Start time is a unix timesatmp, here we convert it to normal time
                    readable_time = datetime.utcfromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')

                    rejected_requests = len(requests) - len(m_commands)
                    sum_accepts += len(m_commands)

                    # print("Total Rejected: ", rejected_requests)
                    sum_rejects += rejected_requests

                    avg_total_dist = np.mean(list(v.travel_dist for index, v in net_v.iterrows()))
                    avg_idle_time = np.mean(list(v.total_idle for index, v in net_v.iterrows()))
                    avg_earnings = np.mean(list(v.earnings for index, v in net_v.iterrows()))
                    avg_cost = np.mean(list(v.cost for index, v in net_v.iterrows()))

                    avg_profit_dqn = np.mean(list(v.earnings - v.cost for index, v in dqn_v.iterrows()))
                    avg_profit_dummy = np.mean(list(v.earnings - v.cost for index, v in dummy_v.iterrows()))

                    # print("P: ", avg_earnings-avg_cost, " P-DQN: ", avg_profit_dqn, " P-D: ", avg_profit_dummy)

                    if average_wt != float(0):
                        average_wt /= len(accepted_commands)

                    summary = "{:s},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}, {:.2f}, {:.2f}".format(
                            readable_time, current_time, len(net_v), len(net_v[net_v.status == status_codes.V_OCCUPIED]), len(requests), len(m_commands),
                            rejected_requests, len(accepted_commands),len(charging_commands), average_wt, avg_earnings, avg_cost, avg_profit_dqn, avg_profit_dummy, avg_total_dist,
                            avg_cap, avg_idle_time)

                    sim_logger.log_summary(summary)

                    if FLAGS.verbose:
                        print("summary: ({})".format(summary), flush=True)

                '''