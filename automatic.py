# %%
import os

ns = [300, 400, 500, 600, 700]

# %%

for i in range(1, 5):
    for n in ns:
        print(f'python gray-scott-mpi.py {n} {i}')
        os.system(f'mpirun -n {i} python3 gray-scott-mpi.py {n}')
# %%


for n in ns:
    os.system(f'python gray-scott.py {n}')