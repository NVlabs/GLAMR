import os
import os.path as osp
import numpy as np
import glob
import datetime
import ctypes
import subprocess
import signal
import itertools


class AverageMeter(object):

    def __init__(self, avg=None, count=1):
        self.reset()
        if avg is not None:
            self.val = avg
            self.avg = avg
            self.count = count
            self.sum = avg * count
    
    def __repr__(self) -> str:
        return f'{self.avg: .4f}'

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if n > 0:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


def worker_init_fn(worker_id):
    os.environ['worker_id'] = str(worker_id)
    np.random.seed(np.random.get_state()[1][0] + worker_id * 7)


def find_last_version(folder, prefix='version_'):
    version_folders = glob.glob(f'{folder}/{prefix}*')
    version_numbers = sorted([int(osp.basename(x)[len(prefix):]) for x in version_folders])
    last_version = version_numbers[-1]
    return last_version


def get_eta_str(cur_iter, total_iter, time_per_iter):
    eta = time_per_iter * (total_iter - cur_iter - 1)
    return convert_sec_to_time(eta)


def convert_sec_to_time(secs):
    return str(datetime.timedelta(seconds=round(secs)))


def set_cuda_l2_cache():
    _libcudart = ctypes.CDLL('libcudart.so')
    # Set device limit on the current device
    # cudaLimitMaxL2FetchGranularity = 0x05
    pValue = ctypes.cast((ctypes.c_int*1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    assert pValue.contents.value == 128


def sync_dir(dir1, dir2, every_n_min=-1, verbose=False):

    cmd = f'rsync -azP {dir1} {dir2}'
    cmd_repeat = f"sh -c 'while true; do {cmd}; sleep {every_n_min}m; done'"

    if every_n_min > 0:
        p = subprocess.Popen(cmd_repeat, shell=True, stdout=None if verbose else subprocess.PIPE, preexec_fn=os.setsid)
        return p
    else:
        subprocess.call(cmd, shell=True, stdout=None if verbose else subprocess.PIPE)
        return None


def kill_shell_process(p):
    os.killpg(os.getpgid(p.pid), signal.SIGTERM)


def concat_lists(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))


def find_consecutive_runs(x, min_len=1):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:] - 1, out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))
        ind = run_lengths >= min_len
        run_starts = run_starts[ind]
        run_lengths = run_lengths[ind]

        # find run values
        run_values = [x[start: start + length] for start, length in zip(run_starts, run_lengths)]
        # assert np.allclose(np.concatenate(run_values), x)

        return run_values, run_starts, run_lengths


if __name__ == '__main__':
    import time

    p = sync_dir('out/', 'out_new', every_n_min=5, verbose=False)
    time.sleep(30)
    kill_shell_process(p)
    