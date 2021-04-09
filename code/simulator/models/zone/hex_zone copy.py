from collections import deque
from simulator.models.charging_pile.charging_pile_repository import ChargingRepository
from simulator.services.routing_service import RoutingEngine
from config.hex_setting import OFF_DURATION
from simulator.models.vehicle.vehicle_repository import VehicleRepository
from dqn_agent.q_network import DeepQNetwork
from simulator.settings import FLAGS
from novelties import agent_codes, status_codes
from simulator.models.customer.customer import Customer
from simulator.models.customer.request import request
from collections import defaultdict
import numpy as np
import contextlib

@contextlib.contextmanager
def local_seed(seed):
    # this defines a local random seed funciton, and let the simulator to resume previous random seed
    state = np.random.get_state()
    np.random.seed(seed)  # set seed
    try:
        yield
    finally:
        np.random.set_state(state)  # put the state back on


class hex_zone:
    def __init__(self,hex_id,coord,coord_list, match_zone,neighbors, charging_station_ids, charging_coords,od_split, trip_time, t_unit, epoch_length):
        '''
        hex_id: id of the hexagon zone in the shapefile
        coord: lon and lat values
        arrival_rate: number of arrivals per tick 
        neighbors: adjacent hexagons' ids
        charging_station_ids: nearest 5 charging station ids
        charging_coords: coordinate of the 5 charging stations
        epoch_length: total ticks per epoch of simulation
        '''
        self.hex_id = hex_id
        self.match_zone_id=match_zone
        self.lon, self.lat = coord
        self.coord_list=coord_list #this is the list for all the lon lat coordinates of the hexagons
        od_split=np.reshape(od_split, (od_split.shape[0],od_split.shape[-1]))
        trip_time = np.reshape(trip_time, (trip_time.shape[0], trip_time.shape[-1]))  #remove one of the dimension

        self.arrival_rate=np.sum(od_split,axis=-1).flatten()/t_unit #now this becomes a  hour by 1 array,and we convert this to each tick of demand!
        #1 by N matrix
        self.od_ratio=od_split
        self.trip_time=trip_time

        #the following two defines the actions
        self.neighbor_hex_id = neighbors # length may vary
        self.nearest_cs = charging_station_ids

        self.passengers = defaultdict()
        self.vehicles = defaultdict()
        self.served_num = 0
        self.veh_waiting_time = 0
        self.rands=[] #the set of random arrivals generated 
        self.served_id=[]
        self.total_pass=0 #this also servers as passenger id

        self.t_unit=t_unit # number of ticks per hour
        self.epoch_length=epoch_length
        self.q_network = DeepQNetwork()
        self.routing_engine = RoutingEngine.create_engine()
        self.cs_loc  =charging_coords
        #initialize the demand for each hexagon zone
        self.init_demand()

    def init_demand(self):
        '''
        todo: generate all the initial demand for each hour. Fix a local random generator to reduce randomness
        :param simulation_length:
        :return:
        '''
        #copy the arrival rate list multiple times!
        with local_seed(self.hex_id):
            self.arrivals = np.random.poisson(list(self.arrival_rate)*int(max(1,np.ceil(self.epoch_length/len(self.arrival_rate)/self.t_unit))),\
                                              size=(self.t_unit, len(self.arrival_rate)*int(max(1,np.ceil(self.epoch_length/len(self.arrival_rate)/self.t_unit)))))
            self.arrivals=self.arrivals.flatten('F') #flatten by columns-major
            self.arrivals=list(self.arrivals)


    def add_veh(self,veh): # vehicle is an object
        '''
        add and remove vehicles by its id
        id contained in veh.state
        :param veh:
        :return:
        '''
        self.vehicles[veh.state.id]=veh
    
    def remove_veh(self,veh):
        self.vehicles.pop(veh.state.id) #remove the vehicle from the list

    def demand_generation(self,tick): #the arrival of passenger demand
        '''
        todo 1: store 60 to config 
        todo 2:[done] complete the demand generation part with travel time and od split ratio
        :param tick: current time
        :return:
        '''
        with local_seed(tick): #fix the random seed
            hour=tick//(self.t_unit*60)%24 #convert into the corresponding hours. Tick are seconds and is incremeted by 60 seconds in each iteration
            # print('hour {}  tick{}'.format(hour, tick))
            narrivals=self.arrivals.pop() #number of arrivals
            destination_rate = self.od_ratio[hour,:]
            if narrivals>0 and sum(destination_rate)>0:
                # print('Tick {} hour {} and tunit{}'.format(tick,hour,self.t_unit))
                destination_rate=destination_rate/sum(destination_rate) #normalize to sum =1
                #lets generate some random des
                destinations=np.random.choice(np.arange(destination_rate.shape[-1]),p=destination_rate,size=narrivals) #choose the destinations
                #request has the information of : id, origin_lon, origin_lat, destination_lon,destination_lat, trip_time
                for i in range(narrivals):
                    # r={'id':self.total_pass,'origin_id':self.hex_id, 'origin_lat':self.lat, 'origin_lon':self.lon, \
                    #    'destination_id':destinations[i], 'destination_lat':self.coord_list[destinations[i]][1], 'destination_lon':self.coord_list[destinations[i]][0], \
                    #        'trip_time':self.trip_time[hour,destinations[i]],'request_time':tick}
                    #r=request(self.total_pass, self.hex_id, (self.lon,self.lat,), destinations[i], self.coord_list[destinations[i]],self.trip_time[hour,destinations[i]],tick)
                    self.passengers[(self.hex_id,self.total_pass)]=Customer(request(self.total_pass, self.hex_id, (self.lon,self.lat,), destinations[i], self.coord_list[destinations[i]],self.trip_time[hour,destinations[i]],tick)) #hex_id and pass_id create a unique passenger identifier
                    self.total_pass+=1

        return

    def demand_generation_async(self,tick): #the arrival of passenger demand
        '''
        this function used for parallel purposes
        :param tick: current time
        :return:
        '''
        newpass=defaultdict()
        with local_seed(tick): #fix the random seed
            hour=tick//(self.t_unit*60)%24 #convert into the corresponding hours. Tick are seconds and is incremeted by 60 seconds in each iteration
            # print('hour {}  tick{}'.format(hour, tick))
            narrivals=self.arrivals.pop() #number of arrivals
            destination_rate = self.od_ratio[hour,:]
            if narrivals>0 and sum(destination_rate)>0:
                # print('Tick {} hour {} and tunit{}'.format(tick,hour,self.t_unit))
                destination_rate=destination_rate/sum(destination_rate) #normalize to sum =1
                #lets generate some random des
                destinations=np.random.choice(np.arange(destination_rate.shape[-1]),p=destination_rate,size=narrivals) #choose the destinations
                #request has the information of : id, origin_lon, origin_lat, destination_lon,destination_lat, trip_time
                for i in range(narrivals):
                    # r={'id':self.total_pass,'origin_id':self.hex_id, 'origin_lat':self.lat, 'origin_lon':self.lon, \
                    #    'destination_id':destinations[i], 'destination_lat':self.coord_list[destinations[i]][1], 'destination_lon':self.coord_list[destinations[i]][0], \
                    #        'trip_time':self.trip_time[hour,destinations[i]],'request_time':tick}
                    #r=request(self.total_pass, self.hex_id, (self.lon,self.lat,), destinations[i], self.coord_list[destinations[i]],self.trip_time[hour,destinations[i]],tick)
                    newpass[(self.hex_id,self.total_pass)]=Customer(request(self.total_pass, self.hex_id, (self.lon,self.lat,), destinations[i], self.coord_list[destinations[i]],self.trip_time[hour,destinations[i]],tick)) #hex_id and pass_id create a unique passenger identifier
                    self.total_pass+=1

        return newpass



    def remove_pass(self,p): #remove passengers
        '''
        Remove passengers by key_id
        :param p:
        :return:
        '''
        self.passengers.pop((p.hex_id,p.id))


    def step(self,tick):
        '''
        Run the step function before after each matching and relocation decision cycle
        this will update the vehicles and passengers
        then generate newly arrived passenger requests
        :param tick:
        :return:
        '''
        #Remove all vehicle in each hex zone as long as they are not idle anymore
        # for v in self.vehicles:
        #     if self.vehicles[v].state.status != status_codes.V_IDLE:
        #         self.remove_veh(self.vehicles[v]) #this can be done outside I guess

        #update unserved passenger waiting time
        '''
        todo: change the following customer picked up code. right now is hard coded: Remove when picked up
        '''
        for pid in self.passengers.keys():
            if self.passengers[pid].status>1:
                self.remove_pass(self.passengers[pid])
            else:
                self.passengers[pid].waiting_time+=1
        
        # perform DQN relocating + charging
        self.dispatch(self.vehicles,tick)
        #arrival of new passengers
        self.demand_generation(tick)

    def dispatch(self,vehicles,current_time):
        # generate dispatch commands, dump transitions to q network
        if len(vehicles) == 0:
            return []
        commands = self.get_dispatch_decisions(vehicles,current_time)
        
        # perform dispatch
        od_pairs = []
        dispatched_veh_list = []
        '''
        todo: finish the logger behind
        '''
        for command in commands:
            dispatched_veh = VehicleRepository.get(command["vehicle_id"])
            if dispatched_veh is None:
                # self.logger.warning("Invalid Vehicle id")
                continue

            if "offduty" in command:
                off_duration = np.random.randint(OFF_DURATION / 2, OFF_DURATION * 3 / 2) # 
                # self.sample_off_duration()   #Rand time to rest
                dispatched_veh.take_rest(off_duration)
            elif "cache_key" in command:
                o_lonlat, d_lonlat = command["cache_key"]
                route, triptime = self.routing_engine.get_route_cache_by_lonlat(o_lonlat, d_lonlat)
            else:
                dispatched_veh_list.append(dispatched_veh)
                od_pairs.append((dispatched_veh.get_location(), command["destination"]))
                
        routes = self.routing_engine.route(od_pairs)
        for dispatched_veh, (route, triptime),command in zip(dispatched_veh_list, routes,commands):
            # if triptime == 0:
            #     continue
            if command['charge_flag'] ==0: # dispatch to relocate
                dispatched_veh.cruise(route, triptime,command['destination_hex_id'],command["action"],current_time)
            else: # to charging station
                c_id = self.nearest_cs[command['action']]
                charging_station_repo = ChargingRepository.get_charging_station(c_id)
                dispatched_veh.head_for_charging_station(command["destination"], triptime,charging_station_repo, route)
        

    def get_dispatch_decisions(self, tbd_vehicles,current_time):
        '''
        :tbd_vehicles: dict of to-be-dispatched vehicles
        '''
        dispatch_commands = []
        a_ids, offduties = self.predict_best_actions(tbd_vehicles,current_time)
        for vid in tbd_vehicles:
            vehicle = tbd_vehicles[vid]# Get best action for this vehicle and whether it will be offduty or not
            if offduties[vid]:
                command = self.create_dispatch_dict(vehicle_id=vehicle.state.vehicle_id, offduty=True,action=a_ids[vid], charge_flag = 0)
            else:
                # Get target destination and key to cache
                target, cache_key, charge_flag, target_hex_id = self.convert_action_to_destination(vehicle, a_ids[vid])
                # create dispatch dictionary with the given attribute
                if target is None:
                    continue
                if cache_key is None:
                    command = self.create_dispatch_dict(vehicle_id= vehicle.state.vehicle_id, destination= target,action=a_ids[vid], charge_flag =charge_flag, dest_hex_id =target_hex_id)
                else:
                    command = self.create_dispatch_dict(vehicle_id=vehicle.state.vehicle_id, cache_key=cache_key,action=a_ids[vid],charge_flag =charge_flag, dest_hex_id =target_hex_id)
            dispatch_commands.append(command)
        return dispatch_commands

    def predict_best_action(self, vehicle,current_time):
        
        if self.q_network is None:
            aidx, offduty = 0, 0
        else:
            # print(vehicle_state.vehicle_id)
            state_rep = [current_time,vehicle.state.vehicle_id,vehicle.state.hex_id,vehicle.state.SOC] # vehicle_state.assigned_hex.get_nearest_cs(), get_cs_waiting_time
            
            aidx = self.q_network.get_action(state=state_rep,num_valid_relo=len([0]+self.neighbor_hex_id))

            offduty = 0
        return aidx, offduty

    def get_state_batches(self, tbd_vehicles,current_time):
        '''
        :vehicles: batches of to-be-dispatched vehicles
        '''
        # if self.q_network is None:
        #     aidx, offduty = 0, 0
        
        state_reps = [[current_time,tbd_vehicles[vid].state.vehicle_id,tbd_vehicles[vid].state.hex_id,tbd_vehicles[vid].state.SOC] for vid in tbd_vehicles]  
        # vehicle_state.assigned_hex.get_nearest_cs(), get_cs_waiting_time
        return state_reps
    def dump_transitions(self,vehicles):
        non_dummy=[vehicles[vid] for vid in vehicles if vehicles[vid].state.type!=agent_codes.dummy_agent]
        transitions = []
        for vehicle in non_dummy:
            if len(vehicle.get_transitions())>0:
                state, action, next_state, reward,flag = vehicle.get_transitions()[-1]
                transitions.append([state, action, next_state, reward,flag])
        return transitions

                    # self.q_network.memory.push(state, action, next_state, reward,flag)
        # self.q_network.train() # start_time = FLAGS.start_time + int(60 * 60 * 24 * FLAGS.start_offset)
        
# Get the destination from dispatched vehicles
    def convert_action_to_destination(self, vehicle, a_id):
        '''
        :to do: 7 is dim of relocation action space
        '''
        cache_key = None
        target = None
        relocation_action_space = [vehicle.state.hex_id]+self.neighbor_hex_id
        
        try:
            target_hex_id = relocation_action_space[a_id]
            lon,lat = self.coord_list[target_hex_id]
            charge_flag = 0
        except IndexError:
            lon,lat = self.cs_loc.loc[self.nearest_cs[a_id-7],['Longitude','Latitude']].values # change 7 later
            vehicle.state.status = status_codes.V_WAYTOCHARGE
            charge_flag = 1
        if lon == vehicle.state.lon and lat == vehicle.state.lat:
            pass
        elif FLAGS.use_osrm and (vehicle.state.lon, vehicle.state.lat) == (lon, lat):
            cache_key = ((vehicle.state.lon, vehicle.state.lat), (lon,lat))
        else:
            target = (lon,lat)

        return target, cache_key, charge_flag, target
    
    def create_dispatch_dict(self, vehicle_id, destination=None, offduty=False, cache_key=None,action = None, charge_flag = 0, dest_hex_id =-1):
            pass
            dispatch_dict = {}
            dispatch_dict["vehicle_id"] = vehicle_id
            dispatch_dict["action"] =action
            dispatch_dict['destination_hex_id'] = dest_hex_id
            
            dispatch_dict["charge_flag"] =charge_flag
            if offduty:
                dispatch_dict["offduty"] = True
            elif cache_key is not None:
                dispatch_dict["cache"] = cache_key
            else:
                dispatch_dict["destination"] = destination
            return dispatch_dict