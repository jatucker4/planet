#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#

import json
import numpy as np
import pickle
import time
import zlib
import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

a = np.random.rand(3, 3, 3)
b = [[1, 1], [2, 2]]

while True:
    #  Wait for next request from client
    message = socket.recv()
    print("Received request: %s" % message)

    #  Do some 'work'
    time.sleep(1)

    #  Send reply back to client
    # socket.send(b"World")
    # stringy = str(203.3)
    # socket.send_string(stringy)

    #a = np.array([1, 1])
    a *= 2
    flags=0
    copy=True
    track=False
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(a.dtype),
        shape = a.shape,
        arr1 = b,
        arr2 = np.ndarray.tolist(a)
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    socket.send(a, flags, copy=copy, track=track)

    # a = np.array([1, 1])
    # flags=0
    # protocol=-1
    # """pickle an object, and zip the pickle before sending it"""
    # p = pickle.dumps(a, protocol)
    # z = zlib.compress(p)
    # socket.send(z, flags=flags)

    #a = np.random.rand(10, 10, 3)
    # a = np.array([1, 10])
    # stringy = str(a)
    # socket.send_string(stringy)



