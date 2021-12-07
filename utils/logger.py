import os
import sys 

class SimulLogger(object):
    '''
    print on terminal and log_name simultaneously
    '''
    def __init__(self, log_name="Default.log"):
        self.terminal = sys.stdout
        self.log_name = log_name
        os.system('rm -rf {}'.format(log_name))

    def write(self, message):
        self.log = open(self.log_name, "a+")
        self.terminal.write(message)
        self.log.write(message)
        self.log.close()

    def flush(self):
        pass
