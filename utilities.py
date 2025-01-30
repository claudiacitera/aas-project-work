from tqdm import tqdm 
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# This script contains utility functions used in the testing phase of the project 
#   - random_test(...) and model_test(...) to test the different agents
#   - plot_training_rewards(...) and plot_cumulative_rewards(...) to plot the rewards obtained respectively after training and testing

def random_test(testing_env):
    '''
    Tests the environment using a random agent.

    Params: 
        env: initialized testing environment
    Returns: 
        testing_history: a list of the rewards obtained in each episode
    '''

    testing_history = [] # List to collect the rewards
    testing_high_score = testing_env.reward_range[0] # Start from the lowest reward
    testing_env.reset()
    episodes = 100

    # I used tqdm to keep track of progress percentages and time spent
    for _ in tqdm(range(episodes)):
        state = testing_env.reset()
        done = False
        testing_episode_score = 0

        while not done:
            action = testing_env.action_space.sample()
            next_state, score, done, _ = testing_env.step(action)
            testing_episode_score += score
            state = next_state

        testing_history.append(testing_episode_score)
        if testing_episode_score > testing_high_score:
            testing_high_score = testing_episode_score

    testing_env.close()

    print(f"Testing Highest Score:{testing_high_score}")
    print(f"Testing Avg Score:{sum(testing_history)/len(testing_history)}")
    print(f"Cumulative Reward over {episodes} episodes: {sum(testing_history)}")
    return testing_history
            
def model_test(agent, testing_env):
    '''
    Tests the environment using a trained agent.

    Params: 
        agent: initialized trained agent,
        env: initialized testing environment
    Returns: 
        testing_history: a list of the rewards obtained in each episode
    '''

    testing_history = [] # List to collect the rewards
    testing_high_score = testing_env.reward_range[0] # Start from the lowest reward
    testing_env.reset()
    episodes = 100

    for _ in tqdm(range(episodes)):
        state = testing_env.reset()
        done = False
        testing_episode_score = 0

        while not done:
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
            action, action_probability, value = agent.sample_action(state_tensor)
            next_state, score, done, _ = testing_env.step(action)
            testing_episode_score += score
            agent.store(state, action, action_probability, value, score, done)
            state = next_state

        testing_history.append(testing_episode_score)
        if testing_episode_score > testing_high_score:
            testing_high_score = testing_episode_score

    testing_env.close()

    print(f"Testing Highest Score:{testing_high_score}")
    print(f"Testing Avg Score:{sum(testing_history)/len(testing_history)}")
    print(f"Cumulative Reward over {episodes} episodes: {sum(testing_history)}")
    return testing_history

def plot_test_rewards(results_dict, game):
    '''
    Plots the cumulative rewards after testing the fully-trained agent, the best agent and a random agent.

    Params: 
        results_dict: a dictionary containing the cumulative rewards obtained by the agents,
        game: the name of the game
    '''
    episodes = range(1, len(results_dict['random']) + 1)
    save_path = f"./results/{game}_rewards.png"

    # I used cumulative sum to get the cumulative reward at the end of episode
    fully_trained_cumulative = np.cumsum(results_dict['fully-trained'])
    best_agent_cumulative = np.cumsum(results_dict['best'])
    random_agent_cumulative = np.cumsum(results_dict['random'])
    
    plt.plot(episodes, fully_trained_cumulative, label='Fully-Trained Agent')
    plt.plot(episodes, best_agent_cumulative, label='Best Agent')
    plt.plot(episodes, random_agent_cumulative, label='Random Agent')
    
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.title(f'Cumulative Reward - {game}')
    plt.legend()
    plt.grid(False)

    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

def plot_training_rewards(rewards, game):
    '''
    Plots the cumulative reward obtained during training.

    Params: 
        rewards: a list of the rewards obtained during training,
        game: the name of the game
    '''
    episodes = range(1, len(rewards) + 1)
    save_path = f"./results/{game}_training_rewards.png"

    training_rewards = np.cumsum(rewards)
    plt.plot(episodes, training_rewards)

    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.title(f'Training Cumulative Reward - {game} ')
    plt.grid(False)

    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()