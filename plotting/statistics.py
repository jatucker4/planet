import numpy as np
import pickle

import json
f = open('timeline.json')
data = json.load(f)
for i in data["traceEvents"]:
    if i["name"] == "graph/collection/train_main_stanford/prev_action_var":
        print(i)
        print(i['ts']/1e6)


# BASE_FOLDER = '00001/test_episodes/' #'072422-2-testing/00001/test_episodes/'
# MAX_STEPS = 200
# ACTION_REPEAT = 1

# episode_dict = pickle.load(open(BASE_FOLDER + "episode_info.p", "rb"))
# rewards = episode_dict['reward']
# steps_taken = episode_dict['steps_taken']

# # Don't count the first episode - it's random
# rewards = np.array(rewards[1:])
# steps_taken = np.array(steps_taken[1:]) * ACTION_REPEAT
# print(steps_taken)
# print("Number of Testing Episodes:", len(steps_taken))
# print("Success Rate:", np.sum([elem < MAX_STEPS - 1 for elem in steps_taken])/len(steps_taken))
# successful_episodes = np.argwhere(steps_taken < MAX_STEPS - 1).flatten()
# rewards = rewards[successful_episodes]
# steps_taken = steps_taken[successful_episodes]
# print("Mean Reward:", np.mean(rewards))
# print("Mean Steps Taken:", np.mean(steps_taken)) 
