import warnings
from humpday.transforms.zcurves import curl_factory

# http://dlib.net/optimization.html#find_min_global
# This library also provides global_function_search which is pretty darn cool

try:
    from dlib import find_min_global, global_function_search, function_spec
    using_dlib = True
except ImportError:
    using_dlib = False

if using_dlib:

    def dlib_default_cube(objective ,n_trials, n_dim, with_count=False):
        global feval_count
        feval_count = 0

        lb = [0. for _ in range(n_dim)]
        ub = [1. for _ in range(n_dim)]

        def _objective(*args) -> float:
            global feval_count
            feval_count += 1
            return objective(list(args))

        best_x, best_val = find_min_global(_objective, lb, ub, n_trials)

        return (best_val, best_x, feval_count) if with_count else (best_val, best_x)

    def dlib_cube(objective ,n_trials, n_dim, with_count):
        return dlib_default_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count)  # It is useful to have a clone of one of the better algos

    def dlib_curl2_cube(objective ,n_trials, n_dim, with_count):
        # Meh
        return curl_factory(optimizer=dlib_default_cube,objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, d=2)

    def dlib_gfs_default_cube(objective, n_trials, n_dim, with_count=False):
        global feval_count
        feval_count = 0
        lb = [0.0 for _ in range(n_dim)]
        ub = [1.0 for _ in range(n_dim)]
        func_spec = function_spec(lb, ub)
        search = global_function_search(func_spec)
        best_val = float('inf')
        best_x = None
        for _ in range(n_trials):
            next_x = search.get_next_x()

            feval_count += 1
            y = objective(list(next_x.x))

            if y < best_val:
                best_val = y
                best_x = list(next_x.x)

            next_x.set(-y)

        return (best_val, best_x, feval_count) if with_count else (best_val, best_x)

    def dlib_gfs_cube(objective, n_trials, n_dim, with_count):
        return dlib_gfs_default_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count)  # It is useful to have a clone of one of the better algos

    def dlib_gfs_curl2_cube(objective, n_trials, n_dim, with_count):
        # Meh
        return curl_factory(optimizer=dlib_gfs_default_cube, objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, d=2)

    DLIB_OPTIMIZERS = [dlib_cube, dlib_default_cube, dlib_gfs_cube, dlib_gfs_default_cube]
    DLIB_TOP_OPTIMIZERS = [dlib_cube, dlib_default_cube, dlib_gfs_cube, dlib_gfs_default_cube]
else:
    DLIB_OPTIMIZERS = []
    DLIB_TOP_OPTIMIZERS = []


if __name__ == '__main__':
    from humpday.objectives.classic import CLASSIC_OBJECTIVES

    for objective in CLASSIC_OBJECTIVES:
        print(' ')
        print(objective.__name__)
        for optimizer in DLIB_OPTIMIZERS:
            print(optimizer(objective, n_trials=500, n_dim=34, with_count=True))
