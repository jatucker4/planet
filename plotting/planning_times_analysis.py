import numpy as np
import pickle

planning_times = pickle.load(open("planning_times.p", "rb"))
# print(planning_times)

planning_times_per_episode = []

# Scroll ahead in the planning_times list to when training begins -- that's when
# batch_env starts to become used
i = 0
while planning_times[i][0] != "batch_env" and i < len(planning_times):
    i += 1
planning_times = planning_times[i:]
print(planning_times)


i = 0
prev_time = planning_times[i][1]
episode_time = 0
num_steps = 0
done = False
while i < len(planning_times):
    batch_env_start = planning_times[i][1]
    stanford_client_start = planning_times[i + 1][1]
    stanford_client_end = planning_times[i + 2][1]
    stanford_client_done = planning_times[i + 3][1]
    

    to_subtract = stanford_client_end - stanford_client_start  # Time spent stepping in environment
    if batch_env_start != prev_time:
        step_time = batch_env_start - prev_time
        step_time -= to_subtract
    else:
        step_time = 0
    
    if not stanford_client_done:
        done = False
        episode_time += step_time  # Time spent planning
        num_steps += 1
    else:
        if not done: # Checking to make sure that this is the first "done" step
            episode_time += step_time  # Time spent planning
            num_steps += 1
            episode_time /= num_steps 
            print("Average planning time for the episode:", episode_time)
            planning_times_per_episode.append(episode_time)
            episode_time = 0  # Reset the episode time and num step counts
            num_steps = 0
            done = True
        else:
            episode_time = 0  # Reset the episode time and num step counts
            num_steps = 0
            done = True

    prev_time = batch_env_start
    i += 4

# print(planning_times_per_episode)
planning_times_per_episode = np.array(planning_times_per_episode)
print("\nAverage planning time over all episodes: ", np.mean(planning_times_per_episode))
