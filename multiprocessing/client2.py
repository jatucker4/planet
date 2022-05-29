#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#

import zmq

context = zmq.Context()

#  Socket to talk to server
print("C2 Connecting to hello world server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

#  Do 10 requests, waiting each time for a response
for request in range(10):
    print("C2 Sending request %s …" % request)
    socket.send(b"C2 Hello")

    #  Get the reply.
    message = socket.recv()
    print("C2 Received reply %s [ %s ]" % (request, message))
    #print(float(message))
