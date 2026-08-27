import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import csv
from collections import deque


RENDER_REFRESH_TIME = 0.02
NON_RENDER_REFRESH_TIME = 0.008


class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNetwork, self).__init__()
        # self.fc1 = nn.Linear(state_dim, 128)
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, action_dim)
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class SatelliteQTableLearner:

    def __init__(self, env, model_name, constraints, reward_csv_filename='episode_rewards.csv'):
        """ Constructor for SatelliteQTableLearner.

        Parameters: 
        env (gym environment): Open AI gym (Toy Text) environment of the game.
        model_name (str): Name of the Open AI gym game

        Returns: None
        """
        self.env = env
        self.model_name = model_name
        action_size = self.env.action_space.n
        state_size = self.env.observation_space.shape[0]
        constraint_size = len(constraints)
        self.constraints = constraints
        self.qtable_constraints = np.zeros((constraint_size, state_size, action_size))
        self.qtable = np.zeros((state_size, action_size))
        self.avg_time = 0
        self.steps_per_episode = 1000
        self.gamma = None

        self.reward_csv_filename = reward_csv_filename

        # write reward to csv
        field_row = ["Episode", "Reward"]
        with open(self.reward_csv_filename, 'a') as f:
            csv_writer = csv.writer(f, delimiter=',', lineterminator='\n')
            csv_writer.writerow(field_row)

    def save_dqn_model(self, policy_net, target_net, optimizer, model_name=None, model_path='saved_models'):
        """Save the DQN model parameters and optimizer state.
    
        Parameters:
            policy_net (DQNetwork): The policy network
            target_net (DQNetwork): The target network
            optimizer (torch.optim): The optimizer
            model_name (str): Name for the saved model files
            model_path (str): Directory to save the model files
        """
        if not model_name:
            model_name = self.model_name

        if not os.path.isdir(model_path):
            os.makedirs(model_path)

        # Save networks and optimizer state
        checkpoint = {
            'policy_net_state_dict': policy_net.state_dict(),
            'target_net_state_dict': target_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
    
        path = os.path.join(model_path, f"{model_name}_dqn.pt")
        torch.save(checkpoint, path)
        print(f'DQN model saved at location: {path}')


    def load_dqn_model(self, state_dim=7, action_dim=None, model_name=None, path="saved_models", evaluate=False):
        """Load a saved DQN model.
    
        Parameters:
            state_dim (int): State dimension
            action_dim (int): Action dimension (if None, uses env.n_action)
            model_name (str): Name of the model to load
            path (str): Directory containing the model
            evaluate (bool): If True, sets model to evaluation mode
    
        Returns:
            tuple: (policy_net, target_net, optimizer)
        """
        if action_dim is None:
            action_dim = self.env.n_action
        
        # Initialize networks
        policy_net = DQNetwork(state_dim, action_dim)
        target_net = DQNetwork(state_dim, action_dim)
        optimizer = optim.Adam(policy_net.parameters())
    
        model_path = os.path.join(path, f"{model_name}.pt")
    
        if not os.path.isfile(model_path):
            print(f'Model file not found at location: {model_path}')
            return policy_net, target_net, optimizer
    
        checkpoint = torch.load(model_path)
    
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        target_net.load_state_dict(checkpoint['target_net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
        if evaluate:
            policy_net.eval()
            target_net.eval()
    
        print(f'[load DQN model]: DQN model loaded from: {model_path}')
        return policy_net, target_net, optimizer


    def round_robin(self, time_interval, n_episode):
        """
        run our proposed RL, Round-Robin and Greedy algorithms should be passed as `learner`
        """
        avg_return = 0
        li = []
        for eps in range(n_episode):
            curr_state = self.env.reset()
            done = False
            iter_num = 0
            total_return = 0
            # this loop should terminate once `done` is True
            while (not done):
                print()
                print("--------------")
                print(curr_state) # current_power, chnl_cap, connected_users, n_drop, sat_pos
                print(f"{len(self.env.env.ue_list)} connecting users")

                # dummy action
                action = (iter_num // time_interval) % self.env.n_action + 25

                new_state, reward, done, info = self.env.step(action)
                curr_state = new_state

                print(f"[iter {iter_num}] reward: {reward}, done: {done}")
                iter_num += 1
                total_return += reward
            avg_return += total_return
            li.append(total_return)
        print(li)
        avg_return /= n_episode
        return avg_return


    def DQN(self, time_interval, n_episode, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.999, # 0.995
        learning_rate=0.001, replay_buffer_size=10000, batch_size=64, target_update_freq=10, 
        save_interval=100, model_name=None, model_path='saved_models', verbose=True):
                                                # 512
        # Get state and action dimensions

        def preprocess_state(state):
            flat_state = []
            for elem in state:
                if isinstance(elem, list):  # If element is a list, flatten it further
                    flat_state.extend(preprocess_state(elem))
                elif isinstance(elem, tuple):  # If element is a tuple, expand it
                    flat_state.extend(elem)
                else:  # Otherwise, append the numeric value directly
                    flat_state.append(elem)
            return flat_state
        

        # new_state [33, 9999.966541062533, 1, 861, 3911784.7182674943, 2076474.8654186632, 5383403.519263018]
        # Note, just hand write. following the above state dimension
        state_dim = 6 # change to spherical coord, only need theta and phi angles
        action_dim = self.env.n_action
        accumulated_rewards = []
        # Initialize networks
        policy_net = DQNetwork(state_dim, action_dim)
        target_net = DQNetwork(state_dim, action_dim)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()  # Target network is not trained directly

        optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        # Replay buffer
        replay_buffer = deque(maxlen=replay_buffer_size)

        # Helper function to select an action
        def select_action(state, epsilon):
            if np.random.rand() < epsilon:
                return np.random.randint(0, action_dim) + 0 # Random action
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                return torch.argmax(q_values).item() + 0

        # Training loop
        avg_return = 0
        li = []  # Store returns for each episode

        for eps in tqdm(range(n_episode), desc="Training Progress"):
            
            curr_state = preprocess_state(self.env.reset())
            total_return = 0  # Accumulated reward for the current episode
            done = False
            iter_num = 0
            
            while not done:
                # Select action using epsilon-greedy policy
                action = select_action(curr_state, epsilon)
                
                # Take action and observe the environment
                new_state, reward, done, info = self.env.step(action)
                new_state = preprocess_state(new_state)
                
                # Store transition in replay buffer
                replay_buffer.append((curr_state, action, reward, new_state, done))

                # Sample a mini-batch from replay buffer
                if len(replay_buffer) >= batch_size:
                    batch = random.sample(replay_buffer, batch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)

                    states = torch.FloatTensor(states)
                    actions = torch.LongTensor(actions).unsqueeze(1)
                    rewards = torch.FloatTensor(rewards).unsqueeze(1)
                    next_states = torch.FloatTensor(next_states)
                    dones = torch.FloatTensor(dones).unsqueeze(1)

                    # Compute Q(s, a) from the policy network
                    q_values = policy_net(states).gather(1, actions)

                    # Compute target Q-values using the target network
                    with torch.no_grad():
                        next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                        target_q_values = rewards + gamma * next_q_values * (1 - dones)

                    # Compute loss and backpropagate
                    loss = loss_fn(q_values, target_q_values)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if verbose:
                        pass

                # Update state and accumulate reward
                curr_state = new_state
                total_return += reward  # Accumulate reward for the current episode
                iter_num += 1

            # Update epsilon (decay exploration)
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            # Update target network periodically
            if eps % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Save model periodically with episode number in filename
            if eps % save_interval == 0 and eps > 0:
                episode_model_name = f"{model_name}_episode_{eps}" if model_name else f"dqn_model_episode_{eps}"
                self.save_dqn_model(policy_net, target_net, optimizer, episode_model_name, model_path)
                # Also save the latest version
                latest_model_name = f"{model_name}_latest" if model_name else "dqn_model_latest"
                self.save_dqn_model(policy_net, target_net, optimizer, latest_model_name, model_path)

            avg_return += total_return
            li.append(total_return)
            # same meaning
            accumulated_rewards.append(total_return)
            # Print accumulated reward for the current episode
            print(f"Episode {eps}, Accumulated Reward: {total_return}, Epsilon: {epsilon}")
            
            # append reward of episode in result csv file
            with open(self.reward_csv_filename, 'a') as f:
                csv_writer = csv.writer(f, delimiter=',', lineterminator='\n')
                csv_writer.writerow([eps, total_return])

            # Save graph every 100 episodes
            if eps % 100 == 0 and eps != 0:
                plt.figure(figsize=(10, 6))
                plt.plot(range(len(accumulated_rewards)), accumulated_rewards, label='Accumulated Reward per Episode')
                plt.xlabel('Episode')
                plt.ylabel('Accumulated Reward')
                plt.title('Accumulated Reward vs. Episode')
                plt.legend()
                plt.grid()
                plt.savefig(f"DQN_reward_episode_{eps}.png")
                plt.close()

        # Save final model
        final_model_name = f"{model_name}_final" if model_name else "dqn_model_final"
        self.save_dqn_model(policy_net, target_net, optimizer, final_model_name, model_path)

        avg_return /= n_episode
        print(f"Average return across all episodes: {avg_return}")


         # Plot the accumulated rewards
        plt.figure(figsize=(10, 6))
        plt.plot(range(n_episode), accumulated_rewards, label='Accumulated Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Accumulated Reward')
        plt.title('Accumulated Reward vs. Episode')
        plt.legend()
        plt.grid()
        plt.show()

    

        return avg_return
