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
dummy = np.array([time[0] for time in planning_times])
print(np.where(dummy == "batch_env"))
print(planning_times[67:75])


# i = 0
# prev_time = planning_times[i][1]
# episode_time = 0
# num_steps = 0
# while i < len(planning_times):
#     # WARNING: Assumes action repeat is 2!
#     batch_env_start = planning_times[i][1]
#     encoder = planning_times[i + 1][1]
#     stanford_client_start1 = planning_times[i + 2][1]
#     stanford_client_end1 = planning_times[i + 3][1]
#     stanford_client_done1 = planning_times[i + 4][1]
#     stanford_client_start2 = planning_times[i + 5][1]
#     stanford_client_end2 = planning_times[i + 6][1]
#     stanford_client_done2 = planning_times[i + 7][1]


#     to_subtract = stanford_client_end1 - stanford_client_start1  # Time spent stepping in environment
#     to_subtract += (stanford_client_end2 - stanford_client_start2)
#     to_subtract += 2 * encoder  # Time spent running the encoder, once in batch_env and once for real
#     if batch_env_start != prev_time:
#         step_time = batch_env_start - prev_time
#         step_time -= to_subtract
#     else:
#         step_time = 0

#     print("STEP TIME", step_time)
#     episode_time += step_time 
#     num_steps += 1

#     if stanford_client_done1:


#     if not stanford_client_done:
#         done = False
#         episode_time += step_time  # Time spent planning
#         num_steps += 1
#     else:
#         if not done: # Checking to make sure that this is the first "done" step
#             episode_time += step_time  # Time spent planning
#             num_steps += 1
#             episode_time /= num_steps 
#             print("Average planning time for the episode:", episode_time)
#             planning_times_per_episode.append(episode_time)
#             episode_time = 0  # Reset the episode time and num step counts
#             num_steps = 0
#             done = True
#         else:
#             episode_time = 0  # Reset the episode time and num step counts
#             num_steps = 0
#             done = True

#     prev_time = batch_env_start
#     i += 8

# # print(planning_times_per_episode)
# planning_times_per_episode = np.array(planning_times_per_episode)
# print("\nAverage planning time over all episodes: ", np.mean(planning_times_per_episode))