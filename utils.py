import os, sys, imghdr

class Logger(object):
    def __init__(self, fp):
        self.file = open(fp, 'a')

    def write(self, str):
        sys.stdout.write(str)
        self.file.write(str)

    def flush(self):
        pass

    def __del__(self):
        self.file.close()


def require_args(opt):
    def require(*args):
        for arg in args:
            assert hasattr(opt, arg) and getattr(opt, arg), \
                'argument --{} required to run --mode={}'.format(arg.replace('_', '-'), opt.mode)
    return require

def summarize_args(opt, verbose):
    def summarize(*args):
        for arg in args:
            print(verbose[arg](getattr(opt, arg))) if arg in verbose and hasattr(opt, arg) else 0
    return summarize


def is_image(path):
    return imghdr.what(path) == 'jpeg' or imghdr.what(path) == 'png'

def list_directory_files(path, policy):
    return [os.path.join(path, f) for f in sorted(os.listdir(path)) if policy(os.path.join(path, f))]

def list_images(path):
    return list_directory_files(path, is_image)
