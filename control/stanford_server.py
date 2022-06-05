import gym
import json
import numpy as np
import pickle
import time
import zlib
import zmq

from planet.control.stanford_client import StanfordEnvironmentClient

#from examples.examples import *  # generate_observation
from planet.humanav_examples.examples import *


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

while True:
    #  Wait for next request from client
    flags=0
    copy=True
    track=False
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    state_arr = np.frombuffer(buf, dtype=md['dtype'])
    state_arr = state_arr.reshape(md['shape'])
    print("Received request for state", state_arr)
    
    #  Send reply back to client
    img = generate_observation_retimg(state_arr)
    flags=0
    copy=True
    track=False
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(img.dtype),
        shape = img.shape
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    socket.send(img, flags, copy=copy, track=track)





