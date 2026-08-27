import gym
from gym import spaces
from wenv.basestation.satellitebasestation import SatelliteBaseStation
from wenv.userequipment.userequipment import UserEquipment
from wenv.environment.environment import Environment
import random
import logging
import numpy as np
import os
import json


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

class CACGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def init_env(self, x_lim, y_lim, terr_parm, sat_parm, n_ue, datarate, ue_data, max_datarate=None, 
                 max_symbols=None, service_class=None, max_power_action=10):
        """init `self.env` as Environment using inputs"""
        self.n_step = 1
        self.env = Environment(x_lim, y_lim, ue_data)
        self.init_pos = []  # for reset method
        
        # random determine user position
        # for i in range(0, n_ue):
        #     pos = (0, 0, 1)
            
        #     service_datarate = datarate
        #     if (service_class is not None):
        #         class_id = self.class_list[i]
        #         service_datarate = service_class[class_id][0] # a tuple containing base DR and level DR
        #     self.datarate = service_datarate

        #     self.env.add_user(
        #         UserEquipment(
        #             self.env, i, service_datarate, pos, speed = 0, direction = 0, #_lambda_c=5, _lambda_d = 15
        #         ))
        #     self.init_pos.append(pos)
        
        for i in range(len(sat_parm)):
            self.env.add_base_station(
                SatelliteBaseStation(
                    env=self.env,
                    bs_id=len(terr_parm) + i,
                    position=sat_parm[i]["pos"],
                    x_y_z=sat_parm[i]["x_y_z"],
                    altitude=sat_parm[i].get("altitude", 1200),  
                    angular_velocity=sat_parm[i].get("angular_velocity", (0, 0)),
                    max_data_rate=max_datarate,
                    max_symbol=max_symbols,
                    max_power_action=max_power_action,  
                ))
        
        self.terr_parm = terr_parm
        self.sat_parm = sat_parm
        

    def __init__(self, x_lim, y_lim, class_list, terr_parm, sat_parm, base_cart, max_cart, datarate = 25, 
                 service_class=None, n_action=3,
                 load_cart_file="../environment/pop_data/user_cart_dict_uniform.json"):
            """
            ### parameters
            - `load_cart_file`: load user coordinate data from json file
            - `datarate`: this member value is filled in `CACGymEnv.init_env` with `service_datarate`
            - `service_class`: service class defined in `my_gym.py`
            - `n_action`: number of possible actions in action space
            - `base_cart`, `max_cart` : carterian coord for the map

            ### state space
            - `power_level` : `self.current_power` stores current power level 
            (note: should set power level higher, at least 10, to actually allocate resource to multi-users)
            - `avg channel capacity` : calculated by subtracting current allocated capacity from total capacity
            - `connected_users` : 

            """
            super(CACGymEnv, self).__init__()
            # elapsed step
            self.n_step = 1

            # QoS
            self.n_drop = 0 # drop count
            self.qos_bonus = 0 # increase for each qos level
            
            self.n_ap = len(terr_parm) + len(sat_parm)
            # also known as power control level
            self.n_action = n_action 

            # set limits on states to control state space size
            self.max_power_level = 10
            self.max_connected_user = 100
            self.bs_max_datarate = 10_000
            self.bs_max_symbols = 10_000

            # RL stuff
            self.action_space = spaces.Discrete(self.n_action)
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6, ), dtype=np.float32) # normalized obs space
            self.a1 = 0.1

            # map bounadry
            self.x_lim = x_lim
            self.y_lim = y_lim
            self.base_cart = base_cart
            self.max_cart = max_cart
            
            # same as the `self.connected_ues` in `satellitebasestation.py`
            self.connected_ues = {}
            self.datarate = datarate
            self.class_list = class_list
            self.n_ue = len(class_list)
            # pick randomly `self.n_next_connecting_ue` users for next step connection
            self.n_next_connecting_ue = 200
            self.service_class = service_class

            # pre-determine class type
            self.load_cart_file = load_cart_file
            bs_dir_name = os.path.dirname(__file__)
            read_json_path = os.path.join(bs_dir_name, self.load_cart_file)
            with open(read_json_path, 'r') as file:
                ue_data = json.load(file)

            self.ue_data = ue_data

            self.init_env(x_lim, y_lim, terr_parm, sat_parm, self.n_ue, datarate, ue_data,
                          max_datarate=self.bs_max_datarate, max_symbols=self.bs_max_symbols, service_class=service_class, 
                          max_power_action=self.max_power_level)
            
            # UE list from satellite BS python file
            self.ue_class_type = {}
            for region, ue_list in ue_data.items():
                for i in range(len(ue_list)):
                    ue_id = f"{region}_{i}"  # Create a unique UE ID (BS.py)
                    self.ue_class_type[ue_id] = self.env.determine_service()


    def observe(self):
        """
        return list of current state for each BS
        - state: current_power, chnl_cap, connected_users, n_drop, sat_pos (normalized)

        output shape: (`self.n_bs`, 5)
        """
        bs_obs = []
        self.n_drop = 0
        total_capacity = self.bs_max_datarate

        for j in range(self.n_ap):
            bs_j = self.env.bs_by_id(j)
            allocated_cap = 0
            ue_allocated_bitrates = bs_j.ue_bitrate_allocation
            zero_bitrate_ue_len = 0
            
            for ue in ue_allocated_bitrates:
                allocated_cap += ue_allocated_bitrates[ue]
                if (ue_allocated_bitrates[ue] <= self.ue_class_type[ue]):
                    self.n_drop += 1
                if (ue_allocated_bitrates[ue] == 0):
                    zero_bitrate_ue_len += 1

            # states
            current_power = bs_j.get_power() # return sat_eirp
            # current_power /= bs_j.get_max_power() # normalize
            chnl_cap = total_capacity - allocated_cap
            # chnl_cap /= total_capacity # normalize

            connected_users = len(ue_allocated_bitrates) - zero_bitrate_ue_len
            connected_users = float(connected_users)
            # connected_users /= float(self.max_connected_user)
            # print(ue_allocated_bitrates)
            n_drop = self.n_drop #/ float(self.max_connected_user)
            # discard z axis
            _, theta, phi = bs_j.get_position() # in spherical coord.
            # normalize
            # print("coord:", theta, phi)
            theta = (float(theta) - 38.0) #/ 4.0
            phi = (float(phi) - 23.7) #/ (39.6 - 23.7)
            sat_pos = (theta, phi)

            curr_obs = [current_power, chnl_cap, connected_users, n_drop, sat_pos]
            # print("\nobs:", curr_obs, "\n")
            bs_obs.append(curr_obs)

        return bs_obs
    
    def is_done(self, sat_pos):
        """
        based on satellite position, judging if `done` in this step

        - `sat_pos`: 3D carteiran coord.
        """
        done = ((sat_pos[1] <= 38) or (sat_pos[1] >= 42) or (sat_pos[2] <= 23.7) or (sat_pos[2] >= 39.6))
            
        return done

    def step(self, action):
        """
        given next power level (action), move on to next time slot
        """
        self.n_step += 1

        # we have only one BS to connect...
        select_bs = 0
        
        bs_j = self.env.bs_by_id(select_bs)
        ue_allocated_bitrates = bs_j.ue_bitrate_allocation

        # compute drop amount
        for ue in ue_allocated_bitrates:
            dr_ue = ue_allocated_bitrates[ue]
            dr_desired_ue = self.env.ue_by_id(ue).data_rate
            if (dr_ue == None) or (dr_ue < dr_desired_ue):
                self.n_drop += 1


        # disconnect all UEs that are not wanting to connect
        for ue_id in range(self.n_ue):
            if ue_id not in self.env.connection_advertisement:
                self.env.ue_by_id(ue_id).disconnect()

        self.env.step()
        
        if (self.n_next_connecting_ue <= len(self.env.ue_list)):
            next_ue_keys = random.sample(list(self.env.ue_list.keys()), self.n_next_connecting_ue)
            next_ue_ids = {}
            for ue in next_ue_keys:
                next_ue_ids[ue] = self.env.ue_list[ue]
        else:
            next_ue_ids = self.env.ue_list

        
        for ue in next_ue_ids:
            self.env.ue_by_id(ue).connect_bs(select_bs)

            # you may check `ue` position by enabling following line:
            # print(self.env.ue_by_id(ue).current_position)

        # set to next power level
        bs_j.set_power_action(action)

        # after the step(), the user that have to appear in the next state is the next user, not the current user
        observation = self.observe()

        # determine if satellite reaches boundary
        connected_user = observation[select_bs][3]
        sat_pos = bs_j.get_position()
        done = self.is_done(sat_pos)

        reward = self.a1 * (self.n_drop // self.n_step) + (1 - self.a1) * bs_j.get_power()
        reward *= -1
        # print()
        # print("reward term:", self.n_drop, bs_j.get_power())
        # print()

        # nothing for `info` for now
        info = {}

        return observation, reward, done, info

    def reset(self):
        self.init_env(self.x_lim, self.y_lim, self.terr_parm, self.sat_parm, self.n_ue, self.datarate, self.ue_data, self.max_power_level)
        # go back 1 time instant, so at the next step() the connection_advertisement list will not change
        observation = self.observe()
        return observation

    def render(self, mode='human'):
        return self.env.render()
    
    # def close (self):
    #     return
