"""
Log to terminal and file, usage:
    import Logger
    Logger.start('logfile.log')
    print('writing to file and console')
    Logger.stop()
"""

import sys

class Logger(object):
    def __init__(self, filename, verbose=True):
        self.terminal = sys.stdout
        self.logfile = open(filename, 'a')
        self.verbose = verbose

    def write(self, message):
        if self.verbose:
            self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        pass

def start(filename, verbose):
    sys.stdout = Logger(filename, verbose)

def stop():
    sys.stdout.logfile.close()
    sys.stdout = sys.stdout.terminal
