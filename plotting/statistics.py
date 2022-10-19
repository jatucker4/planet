import numpy as np
import pickle

BASE_FOLDER = '092422-1-testing-testtrap/00001/test_episodes/' 
MAX_STEPS = 200
NUM_DESIRED_EPISODES = 500

episode_dict = pickle.load(open(BASE_FOLDER + "episode_info.p", "rb"))
rewards = episode_dict['reward']
steps_taken = episode_dict['steps_taken']
step_goal_reached = episode_dict['step_goal_reached']

# Don't count the first episode - it's random
print("Number of Testing Episodes:", len(np.array(rewards[1:])))
rewards = np.array(rewards[1:])[:NUM_DESIRED_EPISODES]
steps_taken = np.array(steps_taken[1:])[:NUM_DESIRED_EPISODES]
step_goal_reached = step_goal_reached[1:][:NUM_DESIRED_EPISODES]
print("STEPS TAKEN", steps_taken)
# print("STEP GOAL REACHED", step_goal_reached)

print("Success Rate:", np.sum([elem < MAX_STEPS - 1 for elem in steps_taken])/len(steps_taken))
successful_episodes = np.argwhere(steps_taken < MAX_STEPS - 1).flatten()
rewards = rewards[successful_episodes]
steps_taken = steps_taken[successful_episodes]
print("Mean Reward:", np.mean(rewards))
print("Mean Steps Taken:", np.mean(steps_taken)) 
