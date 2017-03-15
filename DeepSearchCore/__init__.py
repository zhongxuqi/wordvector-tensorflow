import os
import DBSetup
from Util import LogUtil
import re
from DeepSearchCore import RawTextFormat, WordVectorTrain

BUF_SIZE = 2048

FORMAT_FILENAME = "format_file_out"

ClientCommandsInfo = """Commands:
    help         print the command information
    initDB       initialize database
    format       format raw content
    trainvector  train the word vector
    testvector   test the word vector
"""

def Server(conn):
    client_commands = re.findall("[^ ]+", conn.recv(BUF_SIZE).decode("utf8"))
    if client_commands[0] == "initDB":
        DBSetup.ResetDB()
        conn.send("initialized database.".encode('utf8'))
    elif client_commands[0] == "format":
        if len(client_commands) != 3:
            LogUtil.WriteErr("command format err:" + " ".join(client_commands))
            conn.send("command error.".encode('utf8'))
        elif os.path.isfile(client_commands[1]):
            input_file = open(client_commands[1], "r")
            output_file = None
            if os.path.isdir(client_commands[2]):
                if client_commands[2][-1] == "/":
                    output_file = open(client_commands[2] + FORMAT_FILENAME, "w")
                else:
                    output_file = open(client_commands[2] + "/" + FORMAT_FILENAME, "w")
            elif os.path.isfile(client_commands[2]):
                output_file = open(client_commands[2], "w")
            if output_file is not None:
                conn.send("formatting raw content.".encode('utf8'))
                RawTextFormat.format(input_file, output_file)
                return
            conn.send("output file error.".encode('utf8'))
        else:
            conn.send("input file error.".encode('utf8'))
    elif client_commands[0] == "trainvector":
        if len(client_commands) < 2:
            LogUtil.WriteErr("command format err:" + " ".join(client_commands))
            conn.send("command error.".encode('utf8'))
        elif os.path.isfile(client_commands[1]):
            input_file = open(client_commands[1], "r")
            WordVectorTrain.running(input_file, client_commands[2:], False)
            conn.send("training the word vector.".encode('utf8'))
        else:
            conn.send("input file error.".encode('utf8'))
    elif client_commands[0] == "testvector":
        if len(client_commands) < 2:
            LogUtil.WriteErr("command format err:" + " ".join(client_commands))
            conn.send("command error.".encode('utf8'))
        elif os.path.isfile(client_commands[1]):
            input_file = open(client_commands[1], "r")
            WordVectorTrain.running(input_file, client_commands[2:], True)
            conn.send("training the word vector.".encode('utf8'))
        else:
            conn.send("input file error.".encode('utf8'))
    elif client_commands[0] == "help":
        conn.send(ClientCommandsInfo.encode('utf8'))
    else:
        conn.send(ClientCommandsInfo.encode('utf8'))
