from collections import deque, namedtuple
from os import write
from time import time as now

import cv2

import numpy as np
from mpi4py import MPI
from skimage.morphology import dilation, disk
from skimage.morphology import disk
from skimage.util.dtype import img_as_ubyte
from tqdm import tqdm as progressbar
from contextlib import contextmanager


class Timer:

    def __init__(self, setup): 
        self.setup = setup
        self.times = []
        self.first_time = now()

    def done(self):
        self.times.append(('total', now() - self.first_time))
        with open(f'times/{self.setup}.txt', 'w') as f:
            f.write("('send_recv', 0.0)" + '\n')
            f.write("('gather', 0.0)" + '\n')
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


def gray_scott(u, v, feed_rate, decay_rate,
                   u_diff_rate=0.1, v_diff_rate=0.05,
                   iterations=60, iteration_steps=30):
    
    u = u.copy()
    v = v.copy()

    def laplacian(r):

        r = np.vstack((r[-1], r, r[0]))
        r = np.vstack((r[:, -1], r.T, r[:, 0])).T

        return r[:-2, 1:-1] + r[1:-1, :-2] - 4 * r[1:-1, 1:-1] + r[1:-1, 2:] + r[2:, 1:-1]

    for _ in range(iterations):

        for _ in range(iteration_steps):

            u_laplacian = laplacian(u)
            v_laplacian = laplacian(v)

            reaction_rate = u * v * v

            u += (u_diff_rate * u_laplacian -
                      reaction_rate + feed_rate * (1 - u))

            v += (v_diff_rate * v_laplacian +
                      + reaction_rate - (feed_rate + decay_rate) * v)

        yield u.copy(), v.copy()


def init_reagents(mask):

    u = np.ones_like(mask, dtype='float32')
    v = np.zeros_like(mask, dtype='float32')

    u[mask] = 1.00
    v[mask] = 0.50

    return u, v

def procedure(n):

    setup = {
        'n': n, #512 + 
        'n_frames': 10,
        'steps': 250,
        'preset': 'fingerprints',
        'n_workers': 0
    }

    n = setup['n']
    n_frames = setup['n_frames']
    steps = setup['steps']
    shape = (n, n)
    spawn_probability = 0.00005


    mask = np.random.rand(*shape) < spawn_probability
    mask = dilation(mask, disk(3))

    description = '_'.join([str(setup[key]) for key in setup])

    timer = Timer(description)

    gs = gray_scott(*init_reagents(mask),
                        *presets['fingerprints'], 
                        iterations=n_frames,
                        iteration_steps=steps)

    frames = [u for u, _ in progressbar(gs, total=n_frames)]
    timer.done()

    video_witer = cv2.VideoWriter(
        f'res/{description}.avi',
        cv2.VideoWriter_fourcc(*'XVID'),
        10, (shape[1], shape[0]), 0)

    for frame in frames:
        frame = frame - np.min(frame)
        frame = frame / np.max(frame)
        frame = img_as_ubyte(frame)
        video_witer.write(frame)
    video_witer.release()


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('n', type=int)
args = parser.parse_args()

procedure(args.n)
