import gym
import json
import numpy as np
import pickle
import time
import zlib
import zmq

from planet.control.intermediate_dummy_env_client import IntermediateDummyEnvClient


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

while True:
    #  Wait for next request from client
    message = socket.recv()
    print("Received request: %s" % message)

    #  Send reply back to client
    low = np.zeros([64, 64, 3], dtype=np.float32)
    high = np.ones([64, 64, 3], dtype=np.float32)
    spaces = {'image': gym.spaces.Box(low, high)}
    sp = gym.spaces.Dict(spaces)
    a = sp.sample()
    a = a['image']
    flags=0
    copy=True
    track=False
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(a.dtype),
        shape = a.shape
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    socket.send(a, flags, copy=copy, track=track)

   