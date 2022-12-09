
import os
from os.path import join
import argparse
import datetime
import time
import pathlib
from subprocess import Popen

class JobManager(object):
    interval = 60 * 5
    eps = 20

    def __init__(self, dir_path, job_id, sync_script_path=None):
        self.job_id = job_id 
        self.dir_path = os.path.abspath(dir_path)
        if sync_script_path is not None:
            self.sync_script_path = os.path.abspath(sync_script_path)

        self.ts_dir_path = join(self.dir_path, 'timestamps')
        if not os.path.exists(self.ts_dir_path):
            try:
                os.makedirs(self.ts_dir_path)
            except Exception as e:
                print('[JobManager] Caught Exception:', e.args)
        self.my_ts_path = join(self.ts_dir_path, self.job_id)
        self.my_process = None

    def clear(self):
        current_ts = self._get_current_ts()

        ts_files = os.listdir(self.ts_dir_path)
        for f_name in ts_files:
            f_path = join(self.ts_dir_path, f_name)
            f = pathlib.Path(f_path)
            if not self._check_alive(f_path, current_ts):
                os.remove(f_path)

    def start(self):
        assert self.my_process is None
        self.my_process = Popen(['python3', self.sync_script_path, self.dir_path, self.job_id])
        return True

    def stop(self):
        assert self.my_process is not None
        self.my_process.terminate()
        self.my_process = None

    def check_alive(self, job_id):
        current_ts = self._get_current_ts()
        return self._check_alive(join(self.ts_dir_path, job_id), current_ts)

    def _check_alive(self, path, current_ts):
        f = pathlib.Path(path)
        try:
            print("[DEBUG]")
            print(current_ts - f.stat().st_mtime)
            return current_ts - f.stat().st_mtime <= JobManager.interval + JobManager.eps
        except FileNotFoundError as e:
            return False

    def _get_current_ts(self):
        test_path = join(self.ts_dir_path, 'test')
        with open(test_path, 'w') as f:
            f.write('')
        return pathlib.Path(test_path).stat().st_mtime


def main(dir_path, job_id):
    jman = JobManager(dir_path, job_id, None)
    print("[Sync] start: job_id=", job_id)
    while True:
        with open(jman.my_ts_path, 'w') as f:
            f.write("")
        time.sleep(JobManager.interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_path', type=str)
    parser.add_argument('job_id', type=str)

    args = parser.parse_args()
    main(args.dir_path, args.job_id)
