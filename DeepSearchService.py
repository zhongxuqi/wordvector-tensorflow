import sys, os, signal
import time
import socket
import DeepSearchCore
from Util import LogUtil

commandsInfo = """Commands:
    help       print the command information
    start      start the service
    stop       stop the service
    restart    restart the service
"""

file_path = "/".join(__file__.split("/")[0:-1])
if len(file_path) > 0:
    file_path += "/"
pid_filename = file_path + "deepsearch.pid"
log_filename = file_path + "deepsearch.log"
err_filename = file_path + "deepsearch.err"
HOST = "localhost"
PORT = 17077

def StopDeamon():
    if os.path.exists(pid_filename):
        try:
            pid_file = open(pid_filename, "r")
            pid = pid_file.readline()
            os.kill(int(pid), signal.SIGKILL)
            os.remove(pid_filename)
            return True
        except ValueError as e:
            print(e)
        except ProcessLookupError as e:
            print(e)
            if e.errno == 3:
                os.remove(pid_filename)
                return True
        return False
    return True

def StartDeamon():
    if os.path.exists(pid_filename):
        print("There is a instance running. please stop it first.")
        sys.exit(1)

    try:
        pid = os.fork()
        if pid > 0:
            # exit first parent
            sys.exit(0)
    except OSError as e:
        print("fork #1 failed: %d (%s)"%(e.errno, e.strerror), file=sys.stderr)
        sys.exit(1)
    # decouple from parent environment
    # os.chdir("/")
    os.setsid()
    os.umask(0)
    # do second fork
    try:
        pid = os.fork()
        if pid > 0:
            # exit from second parent, print eventual PID before
            print("Daemon PID %d" % pid)
            pid_file = open(pid_filename, "w")
            pid_file.write(str(pid))
            pid_file.close()
            sys.exit(0)
        main()
    except OSError as e:
        print("fork #2 failed: %d (%s)" % (e.errno, e.strerror), file=sys.stderr)
        sys.exit(1)

service_socket = None
def closeSocket():
    global service_socket
    if service_socket is not None:
        service_socket.close()
        service_socket.shutdown()
    print("success to stop service.")
    sys.exit(0)

def main():
    global service_socket
    signal.signal(signal.SIGTERM, closeSocket)
    signal.signal(signal.SIGTTOU, signal.SIG_IGN)
    signal.signal(signal.SIGTTIN, signal.SIG_IGN)
    signal.signal(signal.SIGTSTP, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    LogUtil.SetOutFiles(open(log_filename, "a"), open(err_filename, "a"))
    try:
        service_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        service_socket.bind((HOST, PORT))
        service_socket.listen(5)
        while True:
            conn, addr = service_socket.accept()
            LogUtil.WriteLog("connect from " + str(addr) + "\n")
            try:
                DeepSearchCore.Server(conn)
            except Exception as e:
                print(e)
            conn.close()
        service_socket.close()
    except OSError as e:
        print(e)
    except TypeError as e:
        print(e)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(commandsInfo)
        sys.exit(1)

    if sys.argv[1] == "start":
        StartDeamon()
    elif sys.argv[1] == "stop":
        if StopDeamon():
            print("success to stop service.")
        else:
            print("fail to stop service.")
    elif sys.argv[1] == "restart":
        if StopDeamon():
            print("success to stop service.")
            StartDeamon()
    else:
        print(commandsInfo)
