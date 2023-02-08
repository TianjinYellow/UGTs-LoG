import os
import torch
import pprint

class OutputManager(object):
    def __init__(self, output_dir, name):
        self.output_dir = output_dir
        self.name = name
        self.save_dir = os.path.join(self.output_dir, name)

        if not os.path.exists(self.output_dir):
            try:
                os.makedirs(self.output_dir)
            except Exception as e:
                print('[OutputManager] Caught Exception:', e.args)

        if not os.path.exists(self.save_dir):
            try:
                os.makedirs(self.save_dir)
            except Exception as e:
                print('[OutputManager] Caught Exception:', e.args)

    def save_dict(self, dic, prefix="dump", ext="pth", name=None):
        filepath = self.get_abspath(prefix, ext, name)
        with open(filepath, 'wb') as f:
            torch.save(dic, f)

    def load_dict(self, prefix="dump", ext="pth", name=None):
        filepath = self.get_abspath(prefix, ext, name)
        return torch.load(filepath)

    def get_abspath(self, prefix, ext, name=None):
        if name is None:
            name = self.name
        return os.path.abspath(os.path.join(self.save_dir, f'{prefix}.{name}.{ext}'))

    def add_log(self):
        pass

    def print(self, *args, prefix=""):
        print(*args)
        print(*args, file=open(os.path.join(self.save_dir, f'{prefix}.{self.name}.out'), "a+"))

    def pprint(self, *args, prefix=""):
        s = pprint.pformat(*args, indent=1)
        self.print(s, prefix=prefix)

if __name__ == '__main__':
    outman = OutputManager('test', 'outman')
    outman.print("a", "b", prefix="thisisprefix")
    outman.print("c", "d", prefix="thisisprefix")
