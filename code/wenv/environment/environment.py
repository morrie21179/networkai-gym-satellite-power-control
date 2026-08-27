import logging
from wenv.userequipment.userequipment import UserEquipment
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import numpy as np
import os
import json

MIN_RSRP = -140

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

class Environment:
    def __init__(self, h, l, ue_data, sampling_time = 1, renderer = None):
        self.h = h
        self.l = l
        self.ue_list = {}
        self.class_list = {}
        self.connection_advertisement = []
        self.bs_list = {}
        self.sampling_time = sampling_time # in seconds
        self.renderer = renderer
        self.plt_run = 0

        self.drone_aps = []
        
        self.ue_data = ue_data # from json file containing user coordinates

        # create dict of user coordinates
        self.ue_positions = {}
        for region, ue_list in ue_data.items():
            for i, ue_pos in enumerate(ue_list):
                ue_id = f"{region}_{i}"  # Create a unique UE ID
                self.ue_positions[ue_id] = ue_pos

        self.n_ue = len(self.ue_positions)
        print(f"[env]: Total {self.n_ue} users")

        """    
        - `required_data_rate`: base level of required data rate (in bps)
        - `qos_level_data_rate`: every rate of this, obtain another qos level (in bps)
        """
        self.service_class = [
                    (100, 100), # BUD traffic
                    (350, 100), # Non real-time video
                    (20, 100), # BAD traffic
                    (0, 0), # No service
                ]
        
        self.all_ue_pos = self.get_all_user_pos()

    def get_all_user_pos(self):
        dir_name = os.path.dirname(os.path.abspath(__file__))
        cart_dict_path = os.path.join(dir_name, "../environment/pop_data/user_cart_dict.json")
        with open(cart_dict_path, 'r') as file:
            ue_data = json.load(file)

        # Combine all UEs from all regions into a single dictionary ##########
        ue_positions = {}
        count = 0
        for _, ue_list in ue_data.items():
            for _, ue_pos in enumerate(ue_list):
                # ue_id = f"{region}_{i}"  # Create a unique UE ID
                ue_id = count
                ue_positions[ue_id] = ue_pos
                count += 1


        return ue_positions
        
    def add_user(self, ue):
        if ue.get_id() in self.ue_list:
            raise Exception("UE ID mismatch for ID %s", ue.get_id())
        self.ue_list[ue.get_id()] = ue
        return
    
    def remove_user(self, ue_id):
        if ue_id in self.ue_list:
            if self.ue_list[ue_id].get_current_bs() != None:
                bs = self.ue_list[ue_id].get_current_bs()
                self.ue_list[ue_id].disconnect()
            # del self.ue_list[ue_id]

    def add_base_station(self, bs):
        if bs.get_id() in self.bs_list:
            raise Exception("BS ID mismatch for ID %s", bs.get_id())
        self.bs_list[bs.get_id()] = bs
        if bs.bs_type == "drone":
            self.drone_aps.append(bs.get_id())
        return

    def compute_rsrp(self, ue):
        rsrp = {}
        for bs in self.bs_list:
            rsrp_i = self.bs_list[bs].compute_rsrp(ue)
            if rsrp_i > MIN_RSRP or self.bs_list[bs].get_bs_type() == "sat":
                rsrp[bs] = rsrp_i
        return rsrp
    
    def advertise_connection(self, ue_id):
        self.connection_advertisement.append(ue_id)
        return
    

    def determine_service(self):
        """
        ### Service definition (TC2 from Table XVI)
        - 0.4 Bursty-User Driven (BUD) traffic (web traffic model) --> 100 Kbps  (60 ~ 1500 KB uniform distribution), every 100 Kbps a Qos level
            - e.g. if allocated 200 Kbps for a BUD traffic user, his QoS level is 2
        - 0.4 Non real-time video                                  --> 35 Mbps (with frame rate of 24 fps), every 10 Mbps a QoS level 
        - 0.075 Bursty-Application Driven (BAD) traffic            --> 2 Mbps, every 10 Mbps a QoS level
        - else no request

        (X) file size -- each follows FTP traffic model

        returns:
        - `class id`: id of the service class
        """
        prob = np.random.uniform(0, 1)
        class_id = 0

        # BUD traffic
        if (prob < 0.4):
            class_id = 0
        # Non real-time video
        elif (0.4 <= prob and prob < 0.8):
            class_id = 1
        # BAD traffic
        elif (0.8 <= prob and prob < 0.875):
            class_id = 2
        # No service
        else:
            class_id = 3

        return class_id
    

    def update_user_pos(self, connected_ue:dict):
        """
        Due to satellite movement, we need to update user list

        - `connected_ue`: (ue_id : ue_pos)
        """
        for ue in self.ue_list:
            self.remove_user(ue)
        self.ue_list = {}
        self.class_list = {}

        for new_ue in connected_ue:
            class_id = self.determine_service()
            self.class_list[new_ue] = class_id
            service_datarate = self.service_class[class_id][0]
            
            self.add_user(
                UserEquipment(
                    self, new_ue, service_datarate, connected_ue[new_ue], speed = 0, direction = 0,# _lambda_c=5, _lambda_d = 15
                ))


    def step(self, substep = False): 
        # substep indicates if it is just a substep or it is a complete step
        # if substep, then the connection advertisement list should not be updated
        # nor the UEs should update their internal timers to decide if it is time to connect/disconnect
        # if not substep:
        #     self.connection_advertisement.clear()

        # update satellite position
        for _, bs in self.bs_list.items():
            if hasattr(bs, 'bs_type') and bs.bs_type == "sat":
                bs.update_position()
                self.update_user_pos(bs.connected_ues)
                # print(f"Environment: Satellite {bs_id} Position: {bs.get_position()}")
                # update covered users
                

        for ue in self.ue_list:
            self.ue_list[ue].step(substep)

        for bs in self.bs_list:
            self.bs_list[bs].step()


    def render(self):
        if self.renderer != None:
            return self.renderer.render(self)

    def bs_by_id(self, id):
        return self.bs_list[id]
    def ue_by_id(self, id):
        return self.ue_list[id]
    def get_sampling_time(self):
        return self.sampling_time
    def get_x_limit(self):
        return self.l
    def get_y_limit(self):
        return self.h
