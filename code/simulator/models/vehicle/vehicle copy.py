from collections import deque
from scipy.spatial.kdtree import KDTree
from novelties import status_codes
from common import geoutils
from .vehicle_state import VehicleState
from .vehicle_behavior import Occupied, Cruising, Idle, Assigned, OffDuty, Waytocharge, Waitpile, Charging, Tobedispatched
from logger import sim_logger
from logging import getLogger
import numpy as np
from simulator.settings import SUPERCHARGING_PRICE ,SUPERCHARGING_TIME,FLAGS,PENALTY_CHARGING_TIME
from common.geoutils import great_circle_distance
from config.hex_setting import SOC_PENALTY,METER_PER_MILE, BETA_COST, BETA_EARNING

class Vehicle(object):
    behavior_models = {
        status_codes.V_IDLE: Idle(),
        status_codes.V_CRUISING: Cruising(),
        status_codes.V_OCCUPIED: Occupied(),
        status_codes.V_ASSIGNED: Assigned(),
        status_codes.V_OFF_DUTY: OffDuty(),
        status_codes.V_WAYTOCHARGE: Waytocharge(),
        status_codes.V_CHARGING: Charging(),
        status_codes.V_WAITPILE: Waitpile(),
        status_codes.V_TOBEDISPATCHED: Tobedispatched()
    }

    def __init__(self, vehicle_state):
        if not isinstance(vehicle_state, VehicleState):
            raise ValueError
        self.state = vehicle_state
        # print(self.state.type, self.state.max_capacity)
        self.__behavior = self.behavior_models[vehicle_state.status]
        self.__customers = []       # A vehicle can have a list of cusotmers
        self.__charging_piles =[]
        self.__customers_ids = []
        self.__charging_station = []
        self.__route_plan = []
        self.earnings = 0
        self.working_time = 0
        # self.epsilon = 5
        # self.first_dispatched = 0
        self.pickup_time = 0
        self.q_action_dict = {}
        self.duration = [0]*len(self.behavior_models)     # Duration for each state
        # self.all_charging_stations = get_processed_charging_piles()
        self.charging_wait = 0
        self.rb_state=[0,0,0,0]
        self.rb_action= 0
        self.rb_next_state=[0,0,0,0]
        self.rb_reward=0
        self.recent_transitions = deque(maxlen=10)
        self.flag = 0
        # df=gpd.read_file('../data/NYC_shapefiles/reachable_hexes.shp') # tagged_cluster_hex './data/NYC_shapefiles/reachable_hexes.shp'
        # self.hex_kdtree= KDTree(df[['lon','lat']])

    # state changing methods
    def step(self,timestep, tick,hex_collections):
        # print(self.state.id, "Loc: ", self.state.lon, self.state.lat)
        # print(self.state.id, "C: ", self.state.current_capacity)
        self.working_time += timestep
        # if self.state.status == status_codes.V_OCCUPIED:
        #     self.duration[status_codes.V_OCCUPIED] += timestep
        # elif self.state.status == status_codes.V_CRUISING:
        #     self.duration[status_codes.V_CRUISING] += timestep
        # elif self.state.status == status_codes.V_OFF_DUTY:
        #     self.duration[status_codes.V_OFF_DUTY] += timestep
        # elif self.state.status == status_codes.V_ASSIGNED:
        #     self.duration[status_codes.V_ASSIGNED] += timestep

        if self.state.status == status_codes.V_IDLE:
            
            self.duration[status_codes.V_IDLE] += timestep


        if self.__behavior.available:
            self.state.idle_duration += timestep
            # self.state.total_idle += timestep
        else:
            self.state.idle_duration = 0

        try:
            self.__behavior.step(self,timestep=timestep, tick=tick) # , tick
        except:
            logger = getLogger(__name__)
            logger.error(self.state.to_msg())
            raise
        # if self.state.current_hex!=self.state.hex_id:
        if self.state.current_hex!=self.state.hex_id:
            #update each vehicle if the new location (current_hex) is different from its current one (hex_id)
            hex_collections[self.state.hex_id].remove_veh(self)
            hex_collections[self.state.current_hex].add_veh(self)
            self.state.hex_id=self.state.current_hex

    def dump_states(self,tick):
        state_rep = [tick,self.state.vehicle_id,self.state.hex_id,self.state.SOC,self.state.status==status_codes.V_OFF_DUTY]
        return state_rep

    def compute_speed(self, route, triptime):
        lats, lons = zip(*route)
        distance = geoutils.great_circle_distance(lats[:-1], lons[:-1], lats[1:], lons[1:])     # Distance in meters
        speed = sum(distance) / triptime
        self.state.travel_dist += sum(distance)
        self.state.SOC -= sum(distance)*METER_PER_MILE/self.get_mile_of_range() # meter to mile
        return speed

    def compute_fuel_consumption(self): # we calculate charging cost now
        return float(self.state.travel_dist * self.state.full_charge_price / (self.state.mile_of_range))
    
    def compute_charging_cost(self,trip_distance): # 30 min for 80% range
        return float(trip_distance*METER_PER_MILE * (SUPERCHARGING_PRICE * SUPERCHARGING_TIME/(self.state.mile_of_range)))

    def compute_profit(self):
        cost = self.compute_fuel_consumption() # there was "()"
        return self.earnings - cost

    def get_transitions(self):
        return self.recent_transitions

    def get_mile_of_range(self):
        return self.state.set_range()
        
    def get_SOC(self):
        return self.state.SOC

    def get_target_SOC(self):
        return self.state.target_SOC

    def cruise(self, route, triptime,hex_id= 0,a = 0, tick=0):
        assert self.__behavior.available
        self.state.current_hex = hex_id
        self.rb_state = [tick,self.get_id(),self.get_hex_id(),self.get_SOC()]
        self.rb_action = a #lon ,lat
        if triptime == 0:
            self.park(tick)
        speed = self.compute_speed(route, triptime)
        self.__reset_plan()
        self.__set_route(route, speed)
        self.__set_destination(route[-1], triptime)
        self.__change_to_cruising()
        self.__log()

    def head_for_customer(self, destination, triptime, customer_id, distance,tick):
        '''
        :destination: lat, lon
        '''
        assert self.__behavior.available
        self.state.SOC -= distance*METER_PER_MILE/self.get_mile_of_range()
        self.state.travel_dist += distance
        self.__reset_plan()
        self.__set_destination(destination, triptime)
        self.state.assigned_customer_id = customer_id
        self.__customers_ids.append(customer_id)
        self.change_to_assigned()
        self.__log()

    def head_for_charging_station(self, triptime, route,cs_coord,cs_id):
        '''
        :destination:
        '''
        assert self.__behavior.available
        self.compute_speed(route,triptime)
        self.__reset_plan()
        self.__set_destination(cs_coord, triptime)
        self.state.assigned_charging_station = cs_id # ID for charging station
        self.__charging_station.append(cs_id)
        self.__change_to_waytocharge()
        self.__log()

    def take_rest(self, duration):
        assert self.__behavior.available
        self.__reset_plan()
        self.state.idle_duration = 0        # Resetting the idle time
        self.__set_destination(self.get_location(), duration)
        self.__change_to_off_duty()
        self.__log()

    def pickup(self, customer):
        # print("Here", self.state.id)
        # assert self.get_hex_id() == customer.get_origin()
        assert self.get_location() == customer.get_origin_lonlat()
        # if not FLAGS.enable_pooling:
        #     # print("Pickup, not pooling!")
        #     customer.ride_on()
        #     self.__customers.append(customer).
        self.state.current_hex = customer.get_origin()
        customer.ride_on()
        self.__customers.append(customer)
        customer_id = customer.get_id()
        self.__reset_plan() # For now we don't consider routes of occupied trip
        self.state.assigned_customer_id = customer_id
        triptime = customer.get_trip_duration()
        self.__set_destination(customer.get_destination_lonlat(), triptime)
        self.__set_pickup_time(triptime)
        self.__change_to_occupied()
        # self.state.current_capacity += 1
        self.__log()

    def dropoff(self,tick):
        # print(self.get_location(), self.state.destination_lat, self.state.destination_lon)
        assert len(self.__customers) > 0
        lenC = len(self.__customers)
        # assert self.get_location() == self.__customers[lenC-1].get_destination_lonlat()
        self.state.current_hex = self.__customers[lenC-1].get_destination_id()
        customer = self.__customers.pop(0)
        customer.get_off()
        self.customer_payment = customer.make_payment(1, self.state.driver_base_per_trip)
        
        self.earnings += self.customer_payment
        trip_distance = great_circle_distance(customer.get_origin()[0], customer.get_origin()[1],
                                                        customer.get_destination()[0],
                                                        customer.get_destination()[1])
        self.state.travel_dist += trip_distance
        self.state.SOC -= trip_distance*METER_PER_MILE/self.get_mile_of_range() # meter to mile
        self.rb_next_state = [tick,self.get_id(),self.state.current_hex,self.get_SOC()]
        self.rb_reward = BETA_EARNING* self.customer_payment - BETA_COST*self.compute_charging_cost(trip_distance) - SOC_PENALTY*(1-self.get_SOC())
        if self.state.status == status_codes.V_OFF_DUTY:
            self.flag = 1
         # specify it
        if self.get_SOC() <0:
            self.rb_reward -= 50 # additional penalty for running out battery
            self.state.SOC = self.state.target_SOC
            self.__set_destination(self.get_location(), PENALTY_CHARGING_TIME) # 45 min for emergency charging
            self.__change_to_cruising()
            self.recent_transitions.append((self.rb_state,self.rb_action,self.rb_next_state,self.rb_reward,self.flag))
            # self.dqn_network.dump_transitions(self.recent_transitions[-1])
            return
        self.recent_transitions.append((self.rb_state,self.rb_action,self.rb_next_state,self.rb_reward,self.flag))
        # self.dump_replay_buffer()
        # self.dqn_network.dump_transitions(self.recent_transitions[-1])
        self.state.current_capacity = 0
        self.__customers_ids = []
        self.__change_to_idle()
        # latest_cust_id = self.state.assigned_customer_id.pop(0)
        self.__reset_plan()
        self.__log()
        # return customer

    def park(self,tick):
        self.rb_next_state = [tick,self.get_id(),self.state.current_hex,self.get_SOC()]
        self.rb_reward = 0
        if self.state.status == status_codes.V_OFF_DUTY:
            self.flag = 1
        self.recent_transitions.append((self.rb_state,self.rb_action,self.rb_next_state,self.rb_reward,self.flag))
        # self.rb_state = self.rb_next_state
        self.__reset_plan()
        self.__change_to_idle()
        self.__log()

    def start_waitpile(self):
        self.__change_to_waitpile()
        self.__log()

    
    def start_charge(self,charging_pile):
        self.__charging_piles.append(charging_pile)
        self.__reset_plan()
        self.state.lat,self.state.lon = charging_pile.get_cp_location()
        self.state.current_hex = charging_pile.get_cp_hex_id()
        self.__change_to_charging()
        self.__log()
    
    def end_charge(self,tick):
        self.state.current_capacity = 0 # current passenger on vehicle
        self.__customers_ids = []
        # self.state.SOC = float(np.random.normal(0.80,0.02)*self.state.mile_of_range)
        charging_pile = self.__charging_piles.pop(0)
        
        # self.dqn_network.dump_transitions(self.recent_transitions[-1])
        self.__change_to_idle()
        self.__reset_plan()
        self.state.lat,self.state.lon = charging_pile.get_cp_location()
        self.state.current_hex = charging_pile.get_cp_hex_id()
        self.rb_next_state = [tick,self.get_id(),self.state.current_hex,self.get_SOC()]
        self.rb_reward = -SOC_PENALTY* (1-self.get_SOC())
        if self.state.status == status_codes.V_OFF_DUTY:
            self.flag = 1
        self.recent_transitions.append((self.rb_state,self.rb_action,self.rb_next_state,self.rb_reward,self.flag))
        

        # print("After Charging:",self.state.SOC)
        self.__log()
    
    def update_location(self, location, route):
        self.state.lat, self.state.lon = location
        self.__route_plan = route

    def update_customers(self, customer):
        # customer.ride_on()
        self.__customers.append(customer)

    def update_time_to_destination(self, timestep):
        dt = min(timestep, self.state.time_to_destination)
        self.duration[self.state.status] += dt
        self.state.time_to_destination -= dt
        if self.state.time_to_destination <= 0:
            self.state.time_to_destination = 0
            self.state.lat = self.state.destination_lat
            self.state.lon = self.state.destination_lon
            return True
        else:
            return False

    # some getter methods
    def get_id(self):
        vehicle_id = self.state.id
        return vehicle_id

    def get_hex_id(self):
        return self.state.hex_id
        

    def get_customers_ids(self):
        return self.__customers_ids

    def get_charging_pile_ids(self):
        return self.__charging_piles_ids
        

    def get_location(self):
        location = self.state.lat, self.state.lon
        return location

    # def get_xy(self):
    #     x,y=convert_lonlat_to_xy(self.state.lon,self.state.lat)
    #     return x,y

    def get_destination(self):
        destination = self.state.destination_lat, self.state.destination_lon
        return destination

    def get_speed(self):
        speed = self.state.speed
        return speed

    def get_agent_type(self):
        return self.state.agent_type

    def get_price_rates(self):
        return [self.state.price_per_travel_m, self.state.price_per_wait_min]

    def reachedCapacity(self):
        if self.state.current_capacity == self.state.max_capacity:
            return True
        else:
            return False

    def get_assigned_customer_id(self):
        customer_id = self.state.assigned_customer_id
        return customer_id
    
    def get_assigned_cs(self):
        return self.state.assigned_charging_station

    def to_string(self):
        s = str(getattr(self.state, 'id')) + " Capacity: " + str(self.state.current_capacity)
        return s

    def print_vehicle(self):
        print("\n Vehicle Info")
        for attr in self.state.__slots__:
            print(attr, " ", getattr(self.state, attr))

        print("IDS::", self.__customers_ids)
        
        print("PILE_IDS::", self.__charging_piles_ids)
        # print(self.state)
        print(self.__behavior)
        for cus in self.__customers:
            cus.print_customer()
        # print(self.__route_plan)
        print("earnings", self.earnings)
        print("working_time", self.working_time)
        print("current_capacity", self.state.current_capacity)
        # print(self.duration)

    def get_route(self):
        return self.__route_plan[:]

    def get_status(self):
        return self.state.status

    def get_total_dist(self):
        return self.state.travel_dist

    def get_idle_duration(self):
        dur = self.working_time - self.duration[status_codes.V_OCCUPIED] - self.duration[status_codes.V_ASSIGNED]
        # print(self.duration)
        return dur

    def get_pickup_time(self):
        return self.pickup_time

    def get_state(self):
        state = []
        for attr in self.state.__slots__:
            state.append(getattr(self.state, attr))
        return state

    def get_score(self):
        score = [self.working_time, self.earnings] + self.duration
        return score

    def get_num_cust(self):
        return self.state.current_capacity

    def get_vehicle(self, id):
        if id == self.state.id:
            return self

    def exit_market(self):
        return False
        # if self.__behavior.available:
        #     if self.state.status == status_codes.V_CHARGING or status_codes.V_CHARGING:
        #         # print("Charging vehicles will not exit")
        #         return False
        #     else:
        #         if self.state.idle_duration == 0:
        #             return self.working_time > MIN_WORKING_TIME # 20 hrs
        #         else:
        #             return self.working_time > MAX_WORKING_TIME # 21 hrs
        # else:
        #     return False

    def __reset_plan(self):
        self.state.reset_plan()
        self.__route_plan = []

    def __set_route(self, route, speed):
        # assert self.get_location() == route[0]
        self.__route_plan = route
        self.state.speed = speed

    def __set_destination(self, destination, triptime):
        self.state.destination_lat, self.state.destination_lon = destination
        self.state.time_to_destination = triptime

    def __set_pickup_time(self, triptime):
        self.pickup_time = triptime

    def __change_to_idle(self):
        self.__change_behavior_model(status_codes.V_IDLE)

    def __change_to_cruising(self):
        self.__change_behavior_model(status_codes.V_CRUISING)

    def change_to_assigned(self):
        self.__change_behavior_model(status_codes.V_ASSIGNED)

    def __change_to_occupied(self):
        self.__change_behavior_model(status_codes.V_OCCUPIED)

    def __change_to_off_duty(self):
        self.__change_behavior_model(status_codes.V_OFF_DUTY)

    def __change_to_waytocharge(self):
        self.__change_behavior_model(status_codes.V_WAYTOCHARGE)
        
    def __change_to_charging(self):
        self.__change_behavior_model(status_codes.V_CHARGING)

    def __change_to_waitpile(self):
        self.__change_behavior_model(status_codes.V_WAITPILE)

    def __change_behavior_model(self, status):
        self.__behavior = self.behavior_models[status]
        self.state.status = status

    def __log(self):
        # self.charging_dict.to_csv("output_charging.csv")
        if FLAGS.log_vehicle:
            sim_logger.log_vehicle_event(self.state.to_msg())
