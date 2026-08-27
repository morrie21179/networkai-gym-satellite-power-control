from json import load
from wenv.basestation.satellitebasestation import SatelliteBaseStation
from wenv.gym.cac_env import CACGymEnv
import logging
import SatelliteQLearning 
import signal
import numpy as np
import os
from wenv.environment.osmnx_test import get_cart
import time
import matplotlib.pyplot as plt

logger = logging.getLogger()
logger.setLevel(level=logging.WARNING)

def exit_handler(signum, frame):
    res = input("Ctrl-c was pressed, do you want to save your current model? y/n ")
    if res == "y":
        global learner
        learner.save_model()
        exit(1)
    else: 
        exit(1)

signal.signal(signal.SIGINT, exit_handler)

"""original evaluation for original `learner`"""
def _evaluate_model(learner, env:CACGymEnv, terr_parm:list, sat_parm:list, quantization:int):
    """
    load model and evaluates

    - `learner`: custom Q table learner. e.g., `lexicographicqlearning.LexicographicQTableLearner`
    - `env`: predefined gym env, e.g., `CACGymEnv`
    - `terr_parm`: list containing terrestrial BS parameters
    - `sat_parm`: satellite BS parameters
    - `quantization`: state space quan.
    """
    #learner.train(train_episodes=100000)
    #learner.save_model()

    learner.load_model("CAC_Env", path="saved_models/100UE_50mbps_5BS_100000_1000/")
    print("Model loaded")
    LQL_rewards = learner.test(test_episodes=1000)
    print("Model tested")

    LL_rewards = ([], [])

    for i in range(1000):
        curr_state = env.reset()
        total_reward = 0
        total_constraint_reward = np.zeros(3)

        for j in range(1000):
            load_levels = np.zeros(len(terr_parm)+len(sat_parm))
            reminder = curr_state
            print(curr_state)

            for k in range(len(load_levels)):
                load_levels[k] = reminder % quantization
                reminder = reminder // quantization

            print(f"Load Level: {load_levels}")
            action = np.argmin(load_levels)
            print(f"Action chosen: {action}")

            new_state, reward, done, info = env.step(action+1)
            curr_state = new_state

            for _ in range(len(info)):
                reward_constr = info[_]

                if reward_constr == -1:
                    reward_constr = 0

                total_constraint_reward[_] += reward_constr

            total_reward += reward

        LL_rewards[0].append(total_reward)
        LL_rewards[1].append(total_constraint_reward)

    np.save("LQL_rewards", LQL_rewards)
    np.save("LL_rewards", LL_rewards)

    print(np.mean(LQL_rewards[0]))
    print(np.mean(LL_rewards[0]))


def run_my_episode(env:CACGymEnv, sat_parm:list, n_episode=1):
    """
    run our proposed RL, Round-Robin and Greedy algorithms should be passed as `learner`
    """

    for eps in range(n_episode):
        curr_state = env.reset()

        # this loop should terminate once `done` is True
        for j in range(200):
            print()
            print("--------------")
            print(curr_state)
            print(f"{len(env.env.ue_list)} connecting users")

            # dummy action
            action = j % env.n_action + 25

            new_state, reward, done, info = env.step(action)
            curr_state = new_state

            print(f"[iter {j}] reward: {reward}, done: {done}")
            

"""    
- `required_data_rate`: base level of required data rate (in bps)
- `qos_level_data_rate`: every rate of this, obtain another qos level (in bps)
"""
SERVICE_CLASS = [
    (100, 100), # BUD traffic
    (350, 100), # Non real-time video
    (20, 100), # BAD traffic
    (0, 0), # No service
]

def determine_service():
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


def main():
    """main function & defintion"""
    # get cart coordinates from real-world map data
    base_cart, max_cart = get_cart()

    # how big is the map
    x_lim = abs(base_cart[0] - max_cart[0])
    y_lim = abs(base_cart[1] - max_cart[1])

    print("map size:", x_lim, y_lim, "in meters")

    # how many user making request
    n_ue = 0

    # UE's positions (defined in `satellitebasestation.py`)
    # ue_positions = {0:(x_0, y_0), 1:(x_1, y_1), ...}

    # their requested services
    class_list = []
    for _ in range(n_ue):
        class_id = determine_service()
        class_list.append(class_id)

    terr_parm = []

    sat_parm = [{
            # "pos": (3591576, 2500219, 0),            # (x, y, 0) 
            "pos": (6971000, 38, 23.7), # (R+h, theta, phi) -> latitude: 90-theta, longitude: phi
            "x_y_z": (3591576, 2500219, 0),
            "altitude": 600000,                      # 300, 600, 1200 km
            "angular_velocity": (0.0222, 0.0883)    # angular velocity
            # "min_elevation_angle": 10,  
        }]

    # define environment
    env = CACGymEnv(x_lim, y_lim, class_list, terr_parm, sat_parm,
                    base_cart=base_cart, max_cart=max_cart, datarate = 50, service_class=SERVICE_CLASS,
                    load_cart_file="../environment/pop_data/user_cart_dict.json")

    # define my learner
    learner = SatelliteQLearning.SatelliteQTableLearner(env, "CAC_Env", [0.075, 0.10, 0.15])
    # rr_return = learner.round_robin(time_interval=1, n_episode=5)
    # print(rr_return)

    qq_learning = learner.DQN(time_interval=1, n_episode=3000, model_name="my_dqn_model", model_path="saved_models", save_interval=100)
    print(qq_learning)


if __name__ == '__main__':
    main()
