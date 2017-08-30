from socket import *

clientSocket = None
PORT_NUM = 1


def init_scoket():
    global clientSocket
    global  PORT_NUM
    HOST = '143.248.135.60'
    PORT = PORT_NUM
    ADDR = (HOST,PORT)
    clientSocket = socket(AF_INET, SOCK_STREAM)
    try:
        clientSocket.connect(ADDR)
    except Exception as e:
        return 'Fail'
    return 'OK'

def close_socket():
    global clientSocket
    clientSocket.close()
    clientSocket = None