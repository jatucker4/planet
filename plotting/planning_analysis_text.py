import numpy as np
import pickle

'''
WARNING: This code assumes the action repeat is 2!
'''
MAX_STEPS = 200
STEP_TIME_THRES = 2.0
ENCODER_THRES = 0.2

planning_time_file = "planning_times.txt"  

planning_times_per_episode = []

with open(planning_time_file, 'r') as f:
# Scroll ahead in the planning_times list to when training begins -- that's when
# batch_env starts to become used
    line = f.readline()
    while line != "batch_env\n":
        line = f.readline()
    # Now line = "batch_env\n"
    i = 0  # Planning times counter, starts at an instance of "batch_env"
    j = 1  # Batch env counter 
    episode_time = 0 
    num_steps = 0
    num_steps_counter = 0 
    done_counter = 0 
    line = f.readline() # Line after "batch_env\n"
    prev_time = float(line)
    batch_env = float(line)
   
    encoder = 0
    stanford_client = 0
    stanford_client_reached_goal = False
    stanford_client_done = False
    line = f.readline() # This will begin 2 lines after "batch_env\n"
    while line != '': 
        if line == "batch_env\n":
            batch_env = float(f.readline())
        elif line == "encoder\n":
            encoder = float(f.readline())
            # if encoder > ENCODER_THRES:
            #     print("\n\n\n ENCODER", num_steps_counter, "\n\n\n")
        elif line == "stanford_client\n":
            stanford_client += float(f.readline())
            num_steps_counter += 1
            done_counter += 1
        elif line == "stanford_client_reached_goal\n":
            boolean = f.readline() == "True\n"
            stanford_client_reached_goal += boolean
        elif line == "stanford_client_done\n":
            boolean = f.readline() == "True\n"
            stanford_client_done += boolean
        elif line == "reset_stanford_client\n":
            stanford_client += float(f.readline())
            done_counter += 1
        
        if batch_env != prev_time:  # Completed a planning step
            to_subtract = 2*encoder  # Time spent running the encoder, once in batch_env and once for real
            to_subtract += stanford_client  # Time spent stepping in environment
            step_time = batch_env - prev_time
            # print(step_time, encoder, stanford_client)
            step_time -= to_subtract
            # if stanford_client < 0.1 or stanford_client > 0.4 and step_time < STEP_TIME_THRES:
            #     print(stanford_client, step_time)
            # print(step_time)

            # Hack -- after every 200 steps the code freezes up and step_time is not accurate!
            # Throw out that data
            # Another time the data is not accurate is when the encoder timing is for some reason too large
            if step_time < STEP_TIME_THRES and encoder < ENCODER_THRES: 
                episode_time += step_time 

            num_steps += 1

            if stanford_client_reached_goal: # End of the episode
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
            encoder = 0
            stanford_client = 0
            stanford_client_reached_goal = False
            stanford_client_done = False




        line = f.readline()

        
planning_times_per_episode = np.array(planning_times_per_episode)
print("\nAverage planning time over all episodes: ", np.mean(planning_times_per_episode))        
