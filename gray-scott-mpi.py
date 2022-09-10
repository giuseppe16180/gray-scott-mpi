from collections import deque, namedtuple
from datetime import time

import cv2

import numpy as np
from mpi4py import MPI
from skimage.morphology import dilation, disk
from skimage.morphology import disk
from skimage.util.dtype import img_as_ubyte
from tqdm import tqdm as progressbar
from contextlib import contextmanager


class Timer:
    def __init__(self, rank, setup):
        self.rank = rank
        if self.rank == 0:
            self.times = []
            self.setup = setup
            self.first_time = MPI.Wtime()

    def start(self):
        if self.rank == 0:
            self.start_time = MPI.Wtime()        

    @contextmanager
    def send_recv(self):
        if self.rank == 0:
            self.start()
            yield
            self.times.append(('send_recv', MPI.Wtime() - self.start_time)) 
        else: yield

    @contextmanager
    def gather(self):
        if self.rank == 0:
            self.start()
            yield
            self.times.append(('gather', MPI.Wtime() - self.start_time))
        else: yield

    def done(self):
        if self.rank == 0:
            self.times.append(('total', MPI.Wtime() - self.first_time))
            with open(f'times/{self.setup}.txt', 'w') as f:
                for time in self.times:
                    f.write(str(time) + '\n')


Preset = namedtuple('Preset', ['feed_rate', 'decay_rate'])

presets = {
    'default': Preset(0.0545, 0.062),
    'worms_and_loops': Preset(0.082,  0.06),  # no
    'mazes': Preset(0.029, 0.057),
    'moving_spots': Preset(0.014, 0.054),
    'fingerprints': Preset(0.037, 0.06),
    'precritical_bubbles': Preset(0.082, 0.059),  # no
    'worms_join_into_maze': Preset(0.046, 0.063),
    'bubbles': Preset(0.098, 0.057)  # no
}


def gray_scott_MPI(comm, timer, u, v, feed_rate, decay_rate,
                   u_diff_rate=0.1, v_diff_rate=0.05,
                   iterations=60, iteration_steps=30):

    height = u.shape[0]
    width = u.shape[1]
    dtype = u.dtype

    curr_rank = comm.rank
    prev_rank = (curr_rank - 1) % comm.size
    next_rank = (curr_rank + 1) % comm.size

    if comm.rank == 0:
        u = u.copy()
        v = v.copy()
    else:
        u = np.empty_like(u)
        v = np.empty_like(v)

    sub_u = np.empty((height // comm.size, width), dtype=dtype)
    sub_v = np.empty((height // comm.size, width), dtype=dtype)

    comm.Scatter(u, sub_u, root=0)
    comm.Scatter(v, sub_v, root=0)

    def laplacian(r):

        with timer.send_recv():
        
            comm.Isend([r[0], MPI.FLOAT], dest=prev_rank)
            comm.Isend([r[-1], MPI.FLOAT], dest=next_rank)

            next_row = np.empty(r[0].shape, dtype=dtype)
            prev_row = np.empty(r[0].shape, dtype=dtype)

            comm.Recv([next_row, MPI.FLOAT], source=next_rank)
            comm.Recv([prev_row, MPI.FLOAT], source=prev_rank)

        r = np.vstack((prev_row, r, next_row))
        r = np.vstack((r[:, -1], r.T, r[:, 0])).T

        return r[:-2, 1:-1] + r[1:-1, :-2] - 4 * r[1:-1, 1:-1] + r[1:-1, 2:] + r[2:, 1:-1]

    for _ in range(iterations):

        for _ in range(iteration_steps):

            u_laplacian = laplacian(sub_u)
            v_laplacian = laplacian(sub_v)

            reaction_rate = sub_u * sub_v * sub_v

            sub_u += (u_diff_rate * u_laplacian -
                      reaction_rate + feed_rate * (1 - sub_u))

            sub_v += (v_diff_rate * v_laplacian +
                      + reaction_rate - (feed_rate + decay_rate) * sub_v)

        with timer.gather():
            comm.Gather(sub_u, u, root=0)
            comm.Gather(sub_v, v, root=0)

        if curr_rank == 0:
            yield u.copy(), v.copy()


def init_reagents(mask):

    u = np.ones_like(mask, dtype='float32')
    v = np.zeros_like(mask, dtype='float32')

    u[mask] = 1.00
    v[mask] = 0.50

    return u, v


def procedure(n):

    comm = MPI.COMM_WORLD

    setup = {
        'n': n, # 384
        'n_frames': 10,
        'steps': 250,
        'preset': 'fingerprints',
        'n_workers': comm.size
    }


    n = setup['n'] 
    n = int(n // comm.size) * comm.size


    n_frames = setup['n_frames']
    steps = setup['steps']
    shape = (n, n)
    spawn_probability = 0.00005

    mask = np.random.rand(*shape) < spawn_probability
    mask = dilation(mask, disk(3))

    description = '_'.join([str(setup[key]) for key in setup])

    timer = Timer(comm.rank, description)

    gs = gray_scott_MPI(comm, timer, *init_reagents(mask),
                        *presets['fingerprints'], 
                        iterations=n_frames,
                        iteration_steps=steps)


    if comm.rank == 0:
        
        video_witer = cv2.VideoWriter(
            f'res/{description}.avi',
            cv2.VideoWriter_fourcc(*'XVID'),
            10, (shape[1], shape[0]), 0)
        
        frames = [u for u, _ in progressbar(gs, total=n_frames)]
        timer.done()

        for frame in frames:
            frame = frame - np.min(frame)
            frame = frame / np.max(frame)
            frame = img_as_ubyte(frame)
            video_witer.write(frame)
        video_witer.release()

    else: deque(gs)



import argparse


parser = argparse.ArgumentParser()
parser.add_argument('n', type=int)
args = parser.parse_args()

procedure(args.n)
