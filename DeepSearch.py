import socket, sys

BUF_SIZE = 2048
HOST = "localhost"
PORT = 17077

if __name__ == "__main__":
    command = " ".join(sys.argv[1:]).encode("utf8")
    if len(command) == 0:
        command = "help".encode("utf8")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    client_socket.send(command)
    print(client_socket.recv(BUF_SIZE).decode("utf8"))
    client_socket.close()
