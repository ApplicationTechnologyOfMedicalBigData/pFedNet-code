import logging
from logging import handlers
import sys
import os
import time

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }

    def __init__(self, filename, printflag=True, level='info', when='D', backCount=3, fmt='%(asctime)s - %(message)s'):
        self.logger = logging.getLogger(filename)
        formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
        self.logger.setLevel(self.level_relations.get(level))
        if printflag:
            sh = logging.StreamHandler(stream=sys.stdout)
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        th.setFormatter(formatter)
        self.logger.addHandler(th)



def server_log(now_path,task=None,model=None):
    
    if model=="init":
        logpath = now_path+'/logs/'
        # 实例化训练日志
        if not os.path.exists(logpath):
            os.makedirs(logpath)

        # build log dirs for every run
        x=os.listdir(logpath)
        countdirs=0
        for i in x:
            if os.path.isdir(logpath+i):
                countdirs+=1
        logpath=logpath+str(countdirs)+"/server/"
        os.makedirs(logpath)
        return logpath
    else:
        # log_time = time.strftime("%Y%m%d%H%M%S",  time.localtime())
        # logname = "federated_log" + "_" + task + "_" + model + "_" + str(log_time) + "_log.txt"
        logname = task+".txt"
        logfile = os.path.join(now_path, logname)
        log = Logger(logfile, level='info')
        return log

def client_log(now_path,task,model):
    # 实例化训练日志
    logpath = now_path+'/logs/'
    if not os.path.exists(logpath):
        os.makedirs(logpath)

    # count dirs num
    x=os.listdir(logpath)
    countdirs=0
    for i in x:
        if os.path.isdir(logpath+i):
            countdirs+=1
    logpath=logpath+str(countdirs-1)+"/"

    log_time = time.strftime("%Y%m%d%H%M%S",  time.localtime())
    logname = os.path.basename(__file__).split(".")[0] + "_" + task + "_" + model + "_" + str(log_time) + "_log.txt"
    logfile = os.path.join(logpath, logname)
    log = Logger(logfile, level='info')
    return log

