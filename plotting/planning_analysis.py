import numpy as np
import pickle

'''
WARNING: This code assumes the action repeat is 2!
'''
MAX_STEPS = 200

planning_times = pickle.load(open("planning_times.p", "rb"))
# print(planning_times)

planning_times_per_episode = []

# Scroll ahead in the planning_times list to when training begins -- that's when
# batch_env starts to become used
i = 0
while i < len(planning_times) and planning_times[i][0] != "batch_env":
    i += 1
planning_times = planning_times[i:]
reached = np.array([time[1] for time in planning_times if time[0] == "stanford_client_reached_goal"])
# print("REACHED", np.where(reached == True), len(np.where(reached == True)[0]))
dones = np.array([time[1] for time in planning_times if time[0] == "stanford_client_done"])
# print("DONES", np.where(dones == True), len(np.where(dones == True)[0]))
# for i in range(100):
#     print(planning_times[i])

i = 0  # Planning times counter, starts at an instance of "batch_env"
j = 1  # Batch env counter 
prev_time = planning_times[i][1] # Starting time
episode_time = 0 
num_steps = 0
num_steps_counter = 0 #-1  # Minus 1 to disregard the reset step
done_counter = 0 #-1  # Minus 1 to disregard the reset step
while i < len(planning_times):
    # print(planning_times[i])
    # Advance batch env counter to the next instance of "batch_env"
    while j < len(planning_times) and planning_times[j][0] != "batch_env":
        j += 1
    # print("i, j", i, j)
    encoder = planning_times[i + 1][1]
    # print(planning_times[i + 1])
    stanford_client = 0
    stanford_client_reached_goal = False
    stanford_client_done = False
    for k in range(i + 2, j):
        # print(planning_times[k])
        if planning_times[k][0] == "stanford_client":
            stanford_client += planning_times[k][1]
            num_steps_counter += 1
            done_counter += 1
            # print("STEPS, DONE", num_steps_counter, done_counter)
            # print("STANFORD CLIENT", stanford_client)
        elif planning_times[k][0] == "stanford_client_reached_goal":
            stanford_client_reached_goal += planning_times[k][1]
            # print("STANFORD CLIENT REACHED GOAL", stanford_client_reached_goal)
        elif planning_times[k][0] == "stanford_client_done":
            stanford_client_done += planning_times[k][1]
            # print("STANFORD CLIENT DONE", stanford_client_done)
        elif planning_times[k][0] == "reset_stanford_client":
            stanford_client += planning_times[k][1]
            done_counter += 1
            # print("\nRESET", stanford_client_reached_goal, stanford_client_done, "\n")
    
    to_subtract = 2*encoder  # Time spent running the encoder, once in batch_env and once for real
    to_subtract += stanford_client  # Time spent stepping in environment
    # TODO (sdeglurkar): how to handle end of the data
    if j < len(planning_times):
        batch_env = planning_times[j][1]
        if batch_env != prev_time:
            step_time = batch_env - prev_time
            step_time -= to_subtract
            # print("STEP TIME", step_time)
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
        print("Average planning time for the episode:", episode_time, "NUM STEPS", num_steps_counter) 
        episode_time = 0  # Reset the episode time and num step counts
        num_steps = 0
        num_steps_counter = 0 #-1
    
    # if stanford_client_done or j >= len(planning_times): 
    if done_counter >= MAX_STEPS - 1: # Done
        if num_steps_counter >= MAX_STEPS - 1: # A full episode
            episode_time /= num_steps  # TODO (sdeglurkar): how to handle action repeat
            planning_times_per_episode.append(episode_time)
            print("Average planning time for the episode DONE:", episode_time, "NUM STEPS", num_steps_counter)
        episode_time = 0  # Reset the episode time and num step counts
        num_steps = 0
        num_steps_counter = 0 #-1
        done_counter = 0 #-1

    prev_time = batch_env
    i = j
    j += 1


# print(planning_times_per_episode)
planning_times_per_episode = np.array(planning_times_per_episode)
print("\nAverage planning time over all episodes: ", np.mean(planning_times_per_episode))




