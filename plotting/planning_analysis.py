import numpy as np
import pickle

'''
WARNING: This code assumes the action repeat is 2!
'''
MAX_STEPS = 200

planning_times = pickle.load(open("planning_times2.p", "rb"))
# print(planning_times)

planning_times_per_episode = []

# Scroll ahead in the planning_times list to when training begins -- that's when
# batch_env starts to become used
i = 0
while i < len(planning_times) and planning_times[i][0] != "batch_env":
    i += 1
planning_times = planning_times[i:]


i = 0  # Planning times counter, starts at an instance of "batch_env"
j = 1  # Batch env counter 
prev_time = planning_times[i][1] # Starting time
episode_time = 0 
num_steps = 0
num_steps_counter = 0 
done_counter = 0 
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
    if len(planning_times[i]) == 3 and len(planning_times[i + 1]) == 3:
        if planning_times[i][2] < 0.4: # TODO (sdeglurkar): Hack!
            pickle = planning_times[i][2] 
        if planning_times[i + 1][2] < 0.4:  # TODO (sdeglurkar): Hack!
            pickle += planning_times[i + 1][2]
        print(pickle)
    else:
        pickle = 0
    for k in range(i + 2, j):
        # print(planning_times[k])
        if planning_times[k][0] == "stanford_client":
            stanford_client += planning_times[k][1]
            num_steps_counter += 1
            done_counter += 1
        elif planning_times[k][0] == "stanford_client_reached_goal":
            stanford_client_reached_goal += planning_times[k][1]
            if len(planning_times[k]) == 3 and planning_times[k][2] < 0.4:  # TODO (sdeglurkar): Hack!
                pickle += planning_times[k][2]
                print(planning_times[k][2])
        elif planning_times[k][0] == "stanford_client_done":
            stanford_client_done += planning_times[k][1]
        elif planning_times[k][0] == "reset_stanford_client":
            stanford_client += planning_times[k][1]
            done_counter += 1
            if len(planning_times[k]) == 3 and planning_times[k][2] < 0.4:  # TODO (sdeglurkar): Hack!
                pickle += planning_times[k][2]
                print(planning_times[k][2])
        # elif planning_times[k][0] == "pickle":
        #     pickle += planning_times[k][1]
            # print(planning_times[k])
    
    to_subtract = 2*encoder  # Time spent running the encoder, once in batch_env and once for real
    print("ENCODER", encoder)
    to_subtract += stanford_client  # Time spent stepping in environment
    print("STANFORD CLIENT", stanford_client)
    to_subtract += 2*pickle # Time spent pickling, once for timing and once for real
    print("PICKLE", pickle)
    # TODO (sdeglurkar): how to handle end of the data
    if j < len(planning_times):
        batch_env = planning_times[j][1]
        if batch_env != prev_time:
            step_time = batch_env - prev_time
            print("STEP TIME BEFORE", step_time)
            step_time -= to_subtract
            print("STEP TIME", step_time)
            if step_time < 0:
                print("\n\n\n\n\n\n\n")
            if step_time > 0 and step_time < 1.0: # TODO (sdeglurkar): Hack!
                episode_time += step_time 
    
    num_steps += 1

    if stanford_client_reached_goal or j >= len(planning_times): # End of the episode
        episode_time /= num_steps  # TODO (sdeglurkar): how to handle action repeat
        planning_times_per_episode.append(episode_time)
        print("Average planning time for the episode:", episode_time, "NUM STEPS", num_steps_counter) 
        episode_time = 0  # Reset the episode time and num step counts
        num_steps = 0
        num_steps_counter = 0 
    
    if done_counter >= MAX_STEPS - 1: # Done
        if num_steps_counter >= MAX_STEPS - 1: # A full episode
            episode_time /= num_steps  # TODO (sdeglurkar): how to handle action repeat
            planning_times_per_episode.append(episode_time)
            print("Average planning time for the episode DONE:", episode_time, "NUM STEPS", num_steps_counter)
        episode_time = 0  # Reset the episode time and num step counts
        num_steps = 0
        num_steps_counter = 0 
        done_counter = 0 

    prev_time = batch_env
    i = j
    j += 1


# print(planning_times_per_episode)
planning_times_per_episode = np.array(planning_times_per_episode)
print("\nAverage planning time over all episodes: ", np.mean(planning_times_per_episode))




