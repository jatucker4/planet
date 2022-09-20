import json
import pprint 

f = open('timeline.json')
data = json.load(f)
list_of_dictionaries = []
for i in data["traceEvents"]:
    # if i["name"] == "graph/collection/train_main_stanford/prev_action_var":
    #     print(i)
    #     print(i['ts']/1e6)
    if "graph/encoder/" in i["name"]:
        list_of_dictionaries.append(i)

pprint.pprint(sorted(list_of_dictionaries, key=lambda x: x['ts']))