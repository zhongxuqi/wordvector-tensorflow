import time

log_file = None
err_file = None

def SetOutFiles(logfile, errfile):
    global log_file, err_file
    log_file = logfile
    err_file = errfile

def WriteLog(content):
    if log_file is not None:
        log_file.write(GetCurrTime() + "    " + content + "\n")

def WriteErr(content):
    if err_file is not None:
        err_file.write(GetCurrTime() + "    " + content + "\n")

def GetCurrTime():
    return time.asctime(time.localtime(time.time()))
