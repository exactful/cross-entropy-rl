import random
from collections import namedtuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_SIZE = 128
LEARNING_RATE = 0.001

BATCH_SIZE = 100
PERCENTILE = 30
GAMMA = 0.9

Episode = namedtuple('Episode', ['reward', 'reward_with_discount', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', ['state', 'action'])

class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (env.observation_space.n, ), dtype=np.float32)

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res

class NeuralNet(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)

def generate_batches_of_episodes(env, net, batch_size, actions_n):
    
    episode_reward = 0.0
    episode_steps = []
    batch = []

    sm = nn.Softmax(dim=1)

    # Reset the environment and capture the current state
    state, _ = env.reset()
    
    while True:
        
        # Use the neural network with random.choice to choose an action
        state_t = torch.FloatTensor([state])
        action_probs_t = sm(net(state_t))
        action_probs = action_probs_t.data.numpy()[0]
        action = np.random.choice(actions_n, p=action_probs)
     
        # Apply a step using the chosen action
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        # Add the reward to the total reward for this episode
        episode_reward += reward

        # Record the state before the action was taken and the action itself
        episode_steps.append(EpisodeStep(state=state, action=action))

        # Check if the episode has ended
        if terminated or truncated:

            # Discount the total episode reward to create variability between episodes
            episode_reward_with_discount = episode_reward * (GAMMA ** len(episode_steps))

            # Record the episode
            batch.append(Episode(reward=episode_reward, reward_with_discount=episode_reward_with_discount, steps=episode_steps))

            # Reset vars
            episode_reward = 0.0
            episode_steps = []
            next_state, _ = env.reset()
            
            if len(batch) == batch_size:

                # Return the batch to the training loop
                yield batch
                batch = []
        
        state = next_state

def filter_batch(batch, percentile):

    # Set a threshold based on the n-th percentile of discounted episode rewards within the batch
    episode_reward_threshold = np.percentile(list(map(lambda s: s.reward_with_discount, batch)), percentile)

    best_episodes = []
    batch_states = []
    batch_actions = []
    
    for episode in batch:
        if episode.reward_with_discount > episode_reward_threshold:
            
            # Add the states and actions from a high performing episode
            batch_states.extend(map(lambda step: step.state, episode.steps))
            batch_actions.extend(map(lambda step: step.action, episode.steps))
            
            best_episodes.append(episode)

    return best_episodes[-500:], torch.FloatTensor(batch_states), torch.LongTensor(batch_actions), episode_reward_threshold

def render_n_steps(env, net, steps_n):
    
    sm = nn.Softmax(dim=1) 
    state, _ = env.reset()

    for i in range(steps_n):

        state_t = torch.FloatTensor([state])

        if net is None:
            # Choose a random step
            action = env.action_space.sample()
        else:
            # Choose a step using the (trained) neural network
            action_probs_t = sm(net(state_t))
            action = np.argmax(action_probs_t.data.numpy()[0])

        state, reward, terminated, truncated, _ = env.step(action)
        
        # Render the step on the display
        env.render()

        if terminated or truncated: state, _ = env.reset()            

if __name__ == "__main__":

    # Render random steps before training
    env = DiscreteOneHotWrapper(gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human"))
    render_n_steps(env, None, 50)

    # Create the environment
    env = DiscreteOneHotWrapper(gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False))

    # Capture environment information
    observation_shape = env.observation_space.shape[0]
    actions_n = env.action_space.n

    # Create the neural network
    net = NeuralNet(observation_shape, HIDDEN_SIZE, actions_n)
    objective = nn.CrossEntropyLoss()
    optimiser = optim.Adam(params=net.parameters(), lr=LEARNING_RATE)

    best_episodes_memory = []

    # Generate batches of episodes and iterate a batch at a time
    for iteration, batch in enumerate(generate_batches_of_episodes(env, net, BATCH_SIZE, actions_n)):

        mean_episode_reward = float(np.mean(list(map(lambda s: s.reward, batch))))
        mean_episode_reward_with_discount = float(np.mean(list(map(lambda s: s.reward_with_discount, batch))))

        # Check the mean reward within the batch
        if mean_episode_reward > 0.8:
            print("Environment solved!")
            break

        # Filter the batch and retain the best states and actions for training
        best_episodes_memory, batch_states_t, batch_actions_t, episode_reward_threshold = filter_batch(best_episodes_memory+batch, PERCENTILE)

        # Skip this iteration if we don't have any data to train on
        if not best_episodes_memory:
            continue

        # Train the neural network
        optimiser.zero_grad()
        action_predictions = net(batch_states_t)
        loss = objective(action_predictions, batch_actions_t)
        loss.backward()
        optimiser.step()
        
        # Report performance
        print(f"{iteration}:\tLoss: {round(loss.item(), 4)}\tMean ep reward: {round(mean_episode_reward, 4)}\tMean ep reward with disc: {round(mean_episode_reward_with_discount, 4)}")

    # Render some steps after training
    env = DiscreteOneHotWrapper(gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human"))
    render_n_steps(env, net, 50)

    # Destroy environment
    env.close()