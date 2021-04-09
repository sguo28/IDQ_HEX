from numpy.core import einsumfunc
from simulator.models.customer.customer_repository import CustomerRepository
from simulator.models.vehicle.vehicle_repository import VehicleRepository
import numpy as np
from novelties import status_codes
from collections import defaultdict
# from config.settings import MAP_WIDTH, MAP_HEIGHT
from simulator.services.routing_service import RoutingEngine
import pandas as pd
from config.hex_setting import  num_reachable_hex
class matching_zone:
    def __init__(self,m_id,hex_zones):
        '''
        m_id: macthing zone id
        hex_zones: the list of hex zone objects
        '''
        self.matching_zone_id = m_id
        self.hex_zones = hex_zones
        self.reject_wait_time = 30*60 # sec
        self.routing_engine = RoutingEngine.create_engine()

    def step(self,tick):
        '''
        Perform the matching here.
        :param tick:
        :return:
        '''
        # get all vehicles and passengers first
        all_pass=self.get_all_passenger()
        all_veh=self.get_all_veh()
        self.matching_algorithms(tick,all_pass,all_veh)

        for h in self.hex_zones:
            h.step(tick)

    def get_all_veh(self):
        '''
        :return: all vehicles in the hex areas inside the matching zone
        '''
        lst=[hex_zone.vehicle for hex_zone in self.hex_zones]
        return set().union(*lst)

    def get_all_passenger(self):
        '''
        :return: all available passengers in the list
        todo: consider sorting the passengers based on their time of arrival?
        todo: add passenger status as matched and unmatched
        '''
        available_pass=[]
        for hex_zone in self.hex_zones:
            available_pass+=[p for p in hex_zone.passengers if p.matched==False] #only those unmatched passengers
        return available_pass

    def get_served_num(self):
        return sum([h.served_num for h in self.hex_zones])
    
    def get_veh_waiting_time(self):
        '''
        todo: this function makes no sense # this is the waiting time for a charging pile
        :return:
        '''
        return sum([h.veh_waiting_time for h in self.hex_zones])

    def matching_algorithms(self,tick, passengers,vehicles):
        '''
        todo: complete the matching algorithm here
        passengers: the set of available Customer objects
        vehicles: the set of vehicle objects
        match available vehicles with passengers
        Change the status for passengers and vehicles
        :return: 
        no return here. We will change the mark for each passengers and drivers as they are matched
        '''

        match_commands = []
        if len(passengers) > 0:
            match_commands = self.match_requests(current_time=tick, vehicles= vehicles,requests=passengers)
        self.update_vehicles(vehicles, match_commands) # output is vehicle, but no use here. 

        for command in match_commands:
            vehicle = vehicles[command["vehicle_id"]]
            vehicle.state.current_capacity += 1
            if vehicle is None:
                # self.logger.warning("Invalid Vehicle id")
                continue
            customer = passengers[command["customer_id"]]
            customer.matched=True
            if customer is None:
                # self.logger.warning("Invalid Customer id")
                continue
            
            triptime = command["duration"]
            vehicle.head_for_customer(customer.get_origin(), triptime, customer.get_id(), command["distance"],tick)
            customer.wait_for_vehicle(triptime)

    def update_vehicles(self, vehicles, match_commands):
        '''
        Make matched vehicles change status to matched.
        In-place update for vehicle objects in the set
        '''
        vehicle_ids = [command["vehicle_id"] for command in match_commands]
        for ids in vehicle_ids:
            vehicles[ids].state.status = status_codes.V_ASSIGNED

    ##### match requests to vehicles ######
    def match_requests(self, current_time, vehicles, requests):
        match_list = []
        n_vehicles = len(vehicles)
        if n_vehicles == 0:
            return match_list

        v_latlon = [[veh.state.lat,veh.state.lon] for veh in vehicles]# vehicles[["lat", "lon","hex_id"]]
        
        V = defaultdict(list)
        vid2coord = {}
        for vid, row in enumerate(v_latlon):
            coord = (row[1], row[0]) # x, y
            vid2coord[vid] = coord
            V[coord].append(vid)
        r_latlon = [[order.request.origin_lat, order.request.origin_lon] for order in requests]
        R = defaultdict(list)
        for rid, row in enumerate(r_latlon):
            coord =(row[1], row[0]) # self.get_coord(row.olon, row.olat)
            R[coord].append(rid)
        # V and R are two statuses: vehicle and request per zone.
        
        for coord in range(num_reachable_hex): # change it later 
            if not R[coord]:
                continue

            target_rids = R[coord]
            candidate_vids = V[coord]
            if len(candidate_vids) == 0:
                continue
            
            T, dist = self.eta_matrix(v_latlon, r_latlon)
            
            assignments = self.assign_nearest_vehicle(candidate_vids,target_rids,T.T, dist.T)
            for vid, rid, tt, d in assignments:
                match_list.append(self.create_matching_dict(vid, rid, tt, d))
                V[vid2coord[vid]].remove(vid)

        return match_list



    # Craeting matching dictionary assciated with each vehicle ID
    def create_matching_dict(self, vehicle_id, customer_id, duration, distance):
        match_dict = {}
        match_dict["vehicle_id"] = vehicle_id
        match_dict["customer_id"] = customer_id
        match_dict["duration"] = duration
        match_dict["distance"] = distance
        return match_dict

    # Returns list of assignments
    def assign_nearest_vehicle(self, ori_ids, dest_ids, T, dist):
        assignments = []
        for di, did in enumerate(dest_ids):
            if len(assignments) >= len(ori_ids):
                break

            # Reuturns the min distance
            oi = T[di].argmin()
            
            # t_queue = ChargingRepository.get_average_wait_time(dest_ids)
            # oi = dist[di].argmin()
            tt = T[di, oi] # - t_queue
            dd = dist[di, oi]
            # print("Chosen t: ", tt)
            # print("Chosen D: ", dd)
            if tt > self.reject_wait_time:
                continue
            oid = ori_ids[oi]

            assignments.append((oid, did, tt, dd))
            T[:, oi] = float('inf')
        return assignments

    def eta_matrix(self, origins_array, destins_array):
        try:
            destins = [(lat, lon) for lat, lon in destins_array.values]
        except AttributeError:
            destins = [(loc[0],loc[1]) for loc in destins_array]
        # destins = [(lat, lon) for lat, lon in destins_array.values]
        origins = [(lat, lon) for lat, lon in origins_array.values]
        # origin_set = list(set(origins))
        origin_set = list(origins)
        latlon2oi = {latlon: oi for oi, latlon in enumerate(origin_set)}
        T, d = np.array(self.routing_engine.eta_many_to_many(origin_set, destins), dtype=np.float32)
        
        T[np.isnan(T)] = float('inf')
        d[np.isnan(d)] = float('inf')
        T = T[[latlon2oi[latlon] for latlon in origins]]
        # print("T: ", T)
        # print("D: ", d.shape)
        return [T, d]