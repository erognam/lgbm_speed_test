import time
import numpy as np
from lightgbm import LGBMRegressor


N = (100000, 250)
ITERS = 500
N_ENSB = 10
N_JOBS = 30 # default: 0

x_train = np.random.rand(N[0],N[1])
y_train = np.random.rand(N[0])

t0 = time.time()
for i in range(N_ENSB):
    print('batch', i)
    model = LGBMRegressor(n_estimators=ITERS, n_jobs=N_JOBS, verbosity=-1, subsample=0.9)
    model.fit(x_train, y_train)
print(f'data size: {x_train.shape}, n_estimators: {ITERS}')
print(f'model training : {time.time()-t0:.1f}s')
