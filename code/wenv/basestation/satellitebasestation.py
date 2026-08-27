# skeleton of Base Station classes
from wenv.basestation.generic import BaseStation
from wenv.pathloss.freespace import FreeSpacePathLoss
from scipy import constants
import logging
import math
import json
from wenv.environment.osmnx_test import get_cart
import os

"""
Reward function

func(state):
output EIRP action
"""
class SatelliteBaseStation(BaseStation):
    def __init__(self, env, bs_id, position, x_y_z,
                altitude=600000, angular_velocity=(0, 0), 
                max_data_rate=None, pathloss=None, max_symbol=None, max_power_action=10, **kwargs):
        """
        parameters
        - `max_data_rate` : total data rate
        - `max_symbol` : total symbol that can be allocated
        """
        self.bs_type = "sat"
        self.carrier_bandwidth = 220 # carrier bandwidth [MHz]
        self.carrier_frequency = 28.4 # frequency [Hz] = 28.4GHz
        
        self.base_sat_eirp = 200 # lowest power of satellite
        self.power_action = 1 # power allocation action
        self.max_power_action = max_power_action # maximum value of action
        self.sat_eirp = self.base_sat_eirp * self.power_action #99 #62 #45.1  # satellite effective isotropic radiated power [dBW] 

        #self.path_loss = 188.4  # path loss [dB]
        self.atm_loss = 0.1  # mean atmospheric loss [dB]
        self.ut_G_T = -9.7  # user terminal G/T [dB/K]

        self.bs_id = bs_id

        self.position = position           # Spherical coordinates
        self.x_y_z = x_y_z                 # Cartesian coordinates
        ##############################################################################
        self.radius_earth = 6371000 # radius of the earth in km
        self.altitude = altitude # altitude of the satellite in km
        self.angular_velocity = angular_velocity
        self.min_elevation_angle = kwargs.get("min_elevation_angle", 10) # Degrees
        ##############################################################################

        self.env = env
        self.frame_length = 120832  # [120832 symbols]
        self.rb_length = 288  # reference burst length, fixed [symbols]
        self.tb_header = 280  # traffic burst header, fixed [symbols]
        self.guard_space = 64  # fixed [symbols]
        self.total_users = 0
        self.frame_duration = 2 # lenght of the frame in milliseconds
        
        # in a frame there are 2 reference burst made of 288 symbols each, 
        # with a guard time of 64 symbols between them and between any other burst
        self.total_symbols = (self.frame_length - self.rb_length * 2 - self.guard_space * 2)  
        if (max_symbol is not None):
            self.total_symbols = max_symbol

        self.frame_utilization = 0  # allocated resources
        
        self.total_bitrate = 10000 # default total
        if max_data_rate is not None:
            self.total_bitrate = max_data_rate
        
        if pathloss is None:
            self.pathloss = FreeSpacePathLoss()

        self.allocated_bitrate = 0
        self.ue_allocation = {}
        self.ue_bitrate_allocation = {}
        
        self.T = 10
        self.resource_utilization_array = [0] * self.T
        self.resource_utilization_counter = 0

        self.load_history = []
        self.data_rate_history = []

        self.connected_ues = {}
    

    def update_position(self):
        """
        Update the position of the satellite using spherical coordinates.
        """
        # bs_dir_name = os.path.dirname(__file__)
        # read_json_path = os.path.join(bs_dir_name, "../environment/pop_data/user_cart_dict.json")
        # with open(read_json_path, 'r') as file:
        #     ue_data = json.load(file)

        # Combine all UEs from all regions into a single dictionary
        # ue_data = self.env.ue_data
        # ue_positions = {}
        # for region, ue_list in ue_data.items():
        #     for i, ue_pos in enumerate(ue_list):
        #         ue_id = f"{region}_{i}"  # Create a unique UE ID
        #         ue_positions[ue_id] = ue_pos
        
        h = self.altitude
        r, theta, phi = self.position
        dtheta, dphi = self.angular_velocity
        x, y, z = self.x_y_z

        # half cone angle between covered area and the Earth's core
        psi = math.acos(h / (h + self.radius_earth) * math.cos(math.radians(self.min_elevation_angle))) - math.radians(self.min_elevation_angle)
        # the coverage area
        Area = 2 * math.pi * (r) ** 2 * (1 - math.cos(math.radians(psi)))
        coverage_radius = math.sqrt(Area / math.pi)
        
        
        time_step = self.env.get_sampling_time()

        # Update spherical coordinates
        theta = (theta + dtheta * time_step) 
        phi   = (phi   + dphi   * time_step)


        # Convert back to Cartesian coordinates for position
        x = r * math.sin(math.radians(theta)) * math.cos(math.radians(phi))
        y = r * math.sin(math.radians(theta)) * math.sin(math.radians(phi))
        z = 0  # Coverage area projected on Earth's surface

        self.position = (r, theta, phi)
        self.x_y_z = (x, y, z)

        if (theta <= 38) or (theta >= 42) or (phi <= 23.7) or (phi >= 39.6):
            print("Satellite out of coverage area, Next Episode !!!")
            self.position = (6971000, 38, 23.7)
            self.x_y_z = (3591576, 2500219, 0)
        else:
            self.position = (r, theta, phi)
            self.x_y_z = (x, y, z)

        # # Compute the coverage center on Earth's surface (Cartesian projection)
        # # coverage_x = self.radius_earth * math.sin(math.radians(theta)) * math.cos(math.radians(phi))
        # # coverage_y = self.radius_earth * math.sin(math.radians(theta)) * math.sin(math.radians(phi))
        # coverage_x = self.radius_earth * math.cos(math.radians(latitude)) * math.cos(math.radians(longitude))
        # coverage_y = self.radius_earth * math.cos(math.radians(latitude)) * math.sin(math.radians(longitude))

        # Store the coverage center and radius
        self.coverage_radius = coverage_radius

        # Reset connected UEs
        ## This dictionary will be used for user update
        self.connected_ues = {}

        # Determine which UEs are within the coverage radius
        for ue_id, ue_pos in self.env.ue_positions.items():
            ue_x, ue_y = ue_pos
            distance = math.sqrt((ue_x - x) ** 2 + (ue_y - y) ** 2)
            if distance <= coverage_radius:
                self.connected_ues[ue_id] = ue_pos
    
        # Debug log
        # print(f"Updated Satellite Position (Spherical): r={r}, theta={theta}, phi={phi}")
        # print(f"Coverage Center: x={x}, y={y}, Radius: {coverage_radius}")
        # print(f"Connected UEs: {list(self.connected_ues.keys())}")
        # print(f"Connected UEs' position: {list(self.connected_ues.values())}")
        # print("The number of UE in every step: ", len(self.connected_ues))


    def set_power_action(self, power_action:int):
        """
        added `power_control` as desired eirp actions
        action space: 1 (low power), 2 (medium power), 3 (full power), ..., `self.max_power_action` (max power)
        """

        self.power_action = min(power_action, self.max_power_action)
        self.sat_eirp = self.power_action * self.base_sat_eirp

    def get_position(self):
        return self.position
    
    def get_cart_position(self):
        """convert speherical `self.position` to cartesian"""
        r, theta, phi = self.get_position()
        # to radian
        theta = math.radians(theta)
        phi = math.radians(phi)
        # formula
        x = self.radius_earth * math.sin(theta) * math.cos(phi)
        y = self.radius_earth * math.sin(theta) * math.sin(phi)
        z = r * math.cos(theta)


        return (x,y,z)
    
    def get_carrier_frequency(self):
        return self.carrier_frequency
    
    def get_bs_type(self):
        return self.bs_type
    
    def get_id(self):
        return self.bs_id
    
    def compute_rsrp(self, ue):
        return self.sat_eirp - self.pathloss.compute_path_loss(ue, self)

    def get_rbur(self):
        return sum(self.resource_utilization_array)/(self.T*self.total_symbols)
    
    def get_power(self):
        return self.sat_eirp
    
    def get_max_power(self):
        return self.max_power_action * self.base_sat_eirp

    def connect(self, ue_id, desired_data_rate, rsrp):

        #IMPORTANT: there must always be a guard space to be added to each allocation. This guard space is included  
        # in the frame utilization but not in the ue_allocation dictionary
        N_blocks, r = self.compute_nsymb_SAT(desired_data_rate, rsrp)
        logging.info("N_blocks = %d - r = %f" %(N_blocks, r))
        
        # check if there is enough bitrate
        # rsrp = eirp - path_loss

        if self.total_bitrate - self.allocated_bitrate <= (r * N_blocks):
            dr = self.total_bitrate - self.allocated_bitrate
            dr = max(dr, 1)
            N_blocks, r = self.compute_nsymb_SAT(dr, rsrp)

        # check if required symbols more than enough
        remain_symbol = self.total_symbols - self.frame_utilization
        requested_symbol = self.tb_header + N_blocks*64 + self.guard_space

        # should be enough symbols...
        # print(f"*[Satellite.connect]: {ue_id} require {requested_symbol} symbols (remain: {remain_symbol})")
        if remain_symbol <= requested_symbol:
            N_blocks = math.floor((remain_symbol - self.guard_space - self.tb_header)/64)
            # print(f"[bs.connect] {ue_id} : {self.tb_header + N_blocks*64 + self.guard_space}, remain: {self.total_symbols - self.frame_utilization}")
            # print(f"N block: {N_blocks}\n")
            # print(f"*[Satellite.connect]: Warning, no more symbol for {ue_id} requesting {requested_symbol} symbols")
            if N_blocks <= 0: # we cannot allocate neither 1 block of 64 symbols
                self.ue_allocation[ue_id] = 0
                self.ue_bitrate_allocation[ue_id] = 0
                return 0

        allocated_bitrate = r * N_blocks
        # print(f"*[Satellite.connect]:{ue_id} allocate bitrate: {allocated_bitrate} (r={r}, N_block={N_blocks})") 
        # print()

        if ue_id not in self.ue_allocation:
            self.ue_allocation[ue_id] = requested_symbol
            self.frame_utilization += requested_symbol
        else:
            self.frame_utilization -= self.ue_allocation[ue_id]
            self.ue_allocation[ue_id] = requested_symbol
            self.frame_utilization += self.ue_allocation[ue_id]

        if ue_id not in self.ue_bitrate_allocation:
            self.ue_bitrate_allocation[ue_id] = allocated_bitrate
            self.allocated_bitrate += allocated_bitrate
        else:
            self.allocated_bitrate -= self.ue_bitrate_allocation[ue_id]
            self.ue_bitrate_allocation[ue_id] = allocated_bitrate
            self.allocated_bitrate += allocated_bitrate

        # print()
        # print(self.ue_bitrate_allocation)
        # print()
        # return allocated data rate (Mbps)
        return allocated_bitrate
    
    def disconnect(self, ue_id):
        self.frame_utilization -= self.ue_allocation[ue_id]
        self.allocated_bitrate -= self.ue_bitrate_allocation[ue_id]
        del self.ue_allocation[ue_id]
        del self.ue_bitrate_allocation[ue_id]

    def update_connection(self, ue_id, desired_data_rate, rsrp):
        self.disconnect(ue_id)
        return self.connect(ue_id, desired_data_rate, rsrp)
    
    def step(self):
        self.resource_utilization_array[self.resource_utilization_counter] = self.frame_utilization
        self.resource_utilization_counter += 1
        if self.resource_utilization_counter % self.T == 0:
            self.resource_utilization_counter = 0
        
        self.load_history.append(self.get_usage_ratio())
        self.data_rate_history.append(self.allocated_bitrate)
    
    def compute_sinr(self, rsrp):
        interference = 0
        for elem in rsrp:
            bs_i = self.env.bs_by_id(elem)
            if elem != self.bs_id and bs_i.get_carrier_frequency() == self.carrier_frequency and bs_i.get_bs_type() == "sat":
                rbur_i = bs_i.get_rbur()
                interference += (10 ** (rsrp[elem]/10))*rbur_i
        thermal_noise = constants.Boltzmann*293.15*self.carrier_bandwidth*1e6
        sinr = (10**(rsrp[self.bs_id]/10))/(thermal_noise + interference)
        logging.info("BS %s -> SINR: %s", self.bs_id, str(10*math.log10(sinr)))
        return sinr
    
    def compute_nsymb_SAT(self, data_rate, rsrp):
        r = self.carrier_bandwidth * 1e6 * math.log2(1 + self.compute_sinr(rsrp))
        r = r / self.frame_length # this is the data rate in [b/s] that is possible to obtains for a single symbol assigned every time frame
        r_64 = r * 64 # we can transmit in blocks of 64 symbols
        n_blocks = math.ceil(data_rate * 1e6 / r_64)
        # print(data_rate*1e6, r_64, n_blocks)
        return n_blocks, r_64/1e6 
    
    def get_usage_ratio(self):
        return self.frame_utilization / self.total_symbols
    
    def get_allocated_data_rate(self):
        return self.allocated_bitrate
