#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#

import json
import numpy as np
import pickle
import zlib
import zmq

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

#  Do 10 requests, waiting each time for a response
for request in range(10):
    print("Sending request %s …" % request)
    socket.send(b"Hello")

    #  Get the reply.
    # message = socket.recv()
    # print("Received reply %s [ %s ]" % (request, message))
    #print(np.fromstring(message))
    #print(float(message))

    flags=0
    copy=True
    track=False
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    a = np.frombuffer(buf, dtype=md['dtype'])
    #a = np.frombuffer(buf, dtype='float')
    print(a.reshape(md['shape']))
    #print(a.reshape((1, 2)))
    print(md['arr2'], md['arr2'] == np.ndarray.tolist(a))

    # flags=0
    # protocol=-1
    # """inverse of send_zipped_pickle"""
    # z = socket.recv(flags)
    # p = zlib.decompress(z)
    # print(pickle.loads(p))
    
