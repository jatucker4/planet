import numpy as np
import pickle

'''
WARNING: This code assumes the action repeat is 2!
'''

planning_times = pickle.load(open("planning_times.p", "rb"))
# print(planning_times)

planning_times_per_episode = []

# Scroll ahead in the planning_times list to when training begins -- that's when
# batch_env starts to become used
i = 0
while i < len(planning_times) and planning_times[i][0] != "batch_env":
    i += 1
planning_times = planning_times[i:]
dones = np.array([time[1] for time in planning_times])
print("DONES", np.where(dones == True))
print(planning_times[-50:])

i = 0  # Planning times counter, starts at an instance of "batch_env"
j = 1  # Batch env counter 
prev_time = planning_times[i][1] # Starting time
episode_time = 0 
num_steps = 0
while i < len(planning_times):
    # Advance batch env counter to the next instance of "batch_env"
    while j < len(planning_times) and planning_times[j][0] != "batch_env":
        j += 1
    encoder = planning_times[i + 1][1]
    stanford_client = 0
    stanford_client_reached_goal = False
    stanford_client_done = False
    for k in range(i + 1, j):
        if planning_times[k][0] == "stanford_client":
            stanford_client += planning_times[k][1]
        elif planning_times[k][0] == "stanford_client_reached_goal":
            stanford_client_reached_goal += planning_times[k][1]
        elif planning_times[k][0] == "stanford_client_done":
            stanford_client_done += planning_times[k][1]
    
    to_subtract = 2*encoder  # Time spent running the encoder, once in batch_env and once for real
    to_subtract += stanford_client  # Time spent stepping in environment
    # TODO (sdeglurkar): how to handle end of the data
    if j < len(planning_times):
        batch_env = planning_times[j][1]
        if batch_env != prev_time:
            step_time = batch_env - prev_time
            step_time -= to_subtract
            print("STEP TIME", step_time)
            episode_time += step_time 
    
    # if j - i == 5: 
    #     # 5 is coming from: "encoder", "stanford_client", 
    #     # "stanford_client_reached_goal", and "stanford_client_done" 
    #     # coming between the two consecutive "batch_env"'s
    #     num_steps += 1
    # else:
    #     num_steps += 2
    num_steps += 1

    if stanford_client_reached_goal or j >= len(planning_times): # End of the episode
        episode_time /= num_steps  # TODO (sdeglurkar): how to handle action repeat
        planning_times_per_episode.append(episode_time)
        print("Average planning time for the episode:", episode_time)
        episode_time = 0  # Reset the episode time and num step counts
        num_steps = 0

    prev_time = batch_env
    i = j
    j += 1


# print(planning_times_per_episode)
planning_times_per_episode = np.array(planning_times_per_episode)
print("\nAverage planning time over all episodes: ", np.mean(planning_times_per_episode))


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

