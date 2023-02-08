
import random
import time
import json
from os.path import join, abspath, exists
from os import makedirs
import pprint

from commands.train import train
from utils.sync_jobs import JobManager
from utils.filelock import FileLock, Timeout

pp = pprint.PrettyPrinter(indent=1)

def _check_job_running(jobman, sync_dir, name):
    coordinate_path = abspath(join(sync_dir, 'coordinate.json'))
    with open(coordinate_path, 'r+') as f:
        body = f.read()
        lis = json.loads(body)
        for i, (job_name, status, job_id, dic) in enumerate(lis):
            if job_name == name:
                return jobman.check_alive(job_id)

def _next_job(jobman, sync_dir, my_id):
    coordinate_path = abspath(join(sync_dir, 'coordinate.json'))
    lock_path = abspath(join(sync_dir, 'coordinate.json.lock'))
    lock = FileLock(lock_path, timeout=1)

    with lock:
        if not exists(coordinate_path):
            with open(coordinate_path, "w") as f:
                f.write('[]')
                return None, None

    with lock:
        with open(coordinate_path, 'r+') as f:
            body = f.read()
            lis = json.loads(body)
            print('\n[Parallel] All hyperparams:')
            pp.pprint(lis)
            print('')
            for i, (job_name, status, job_id, dic) in enumerate(lis):
                if status == "not completed":
                    if not jobman.check_alive(job_id):
                        lis[i] = (job_name, status, my_id, dic)
                        f.seek(0)
                        f.write(json.dumps(lis))
                        f.truncate()
                        return job_name, dic
                elif status == "completed":
                    pass
                else:
                    raise NotImplementedError
    return None, None

def _update_job(sync_dir, job_name, status, job_id, dic):
    coordinate_path = abspath(join(sync_dir, 'coordinate.json'))
    assert exists(coordinate_path)

    lock_path = abspath(join(sync_dir, 'coordinate.json.lock'))
    lock = FileLock(lock_path, timeout=1)
    with lock:
        with open(coordinate_path, 'r+') as f:
            lis = json.loads(f.read())
            for i, (name, _, _, _) in enumerate(lis):
                if name == job_name:
                    lis[i] = (job_name, status, job_id, dic)
                    f.seek(0)
                    f.write(json.dumps(lis))
                    f.truncate()
                    return
            lis.append((job_name, status, job_id, dic))
            f.seek(0)
            f.write(json.dumps(lis))
            f.truncate()

def _count_jobs(sync_dir):
    coordinate_path = abspath(join(sync_dir, 'coordinate.json'))
    assert exists(coordinate_path)

    with open(coordinate_path, 'r') as f:
        lis = json.loads(f.read())
        return len(lis)

def parallel(exp_name, cfg):
    sync_dir = cfg['sync_dir']
    sync_dir = abspath(join(sync_dir, exp_name))
    if not exists(sync_dir):
        try:
            makedirs(sync_dir)
        except Exception as e:
            print('[Parallel] Caught Exception:', e.args)

    now = time.time()
    job_id = str(now)
    search_space = cfg['parallel_grid']
    assert search_space is not None

    jobman = JobManager(sync_dir, job_id, sync_script_path="utils/sync_jobs.py")
    jobman.clear()
    jobman.start()

    completed_dics = []
    job_counter = 0
    no_job_counter = 0
    parallel_counter = 0
    max_jobs = 1
    max_no_job = 100
    max_parallel = 1
    for k, cands in search_space.items():
        max_jobs *= len(cands)

    while True:
        if job_counter >= max_jobs:
            print("[Parallel] Stop searching because we execute all patterns")
            break
        if no_job_counter >= max_no_job:
            if parallel_counter >= max_parallel:
                print("[Parallel] Stop searching because searched for enough time")
                break
            else:
                print("[Parallel] Wait for 6 minutes...")
                time.sleep(60 * 6)

            no_job_counter = 0
            parallel_counter += 1

        # resume to train existing hyperparameters
        while True:
            resume_job_name, dic = _next_job(jobman, sync_dir, job_id)
            if resume_job_name is not None:
                for k in dic:
                    cfg[k] = dic[k]
                print('[Parallel] Resume job:', resume_job_name)
                try:
                    train(exp_name, cfg, prefix=resume_job_name)
                    _update_job(sync_dir, resume_job_name, 'completed', job_id, dic)

                    completed_dics.append(dic.copy())
                    job_counter += 1
                    no_job_counter = 0
                    parallel_counter = 0
                except:
                    time.sleep(10)
            else:
                break

        # search for new hyperparameters
        print("[Parallel] Search new hyperparamter...")
        random.seed()
        sampled_dic = {}
        for k, cands in search_space.items():
            sampled_dic[k] = random.sample(cands,k=1)[0]

        # train with new hyperparameters
        job_name = ""
        for k, v in sampled_dic.items():
            job_name += f"{k}_{v}--"
            cfg[k] = v

        if _check_job_running(jobman, sync_dir, job_name):
            no_job_counter += 1
        elif sampled_dic not in completed_dics:
            _update_job(sync_dir, job_name, 'not completed', job_id, sampled_dic)
            print('[Parallel] Start job:', job_name)
            train(exp_name, cfg, prefix=job_name)
            _update_job(sync_dir, job_name, 'completed', job_id, sampled_dic)

            completed_dics.append(sampled_dic.copy())
            job_counter += 1
            no_job_counter = 0
            parallel_counter = 0
        else:
            no_job_counter += 1

    jobman.stop()

