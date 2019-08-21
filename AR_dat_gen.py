import numpy as np
import os
np.random.seed(1)


def data_gen(T, impute, x0, theta, obs_std, dat_dir=os.getcwd()):

    if not os.path.exists(dat_dir + "/dat/"):
        os.makedirs(dat_dir + "/dat/")

    X = np.zeros(np.int32(T + 1))
    X[0] = x0
    for i in range(1, X.shape[0]):
        X[i] = np.random.normal(X[i - 1] * theta[1] + theta[0], theta[2])
    obs = np.random.normal(loc=X, scale=obs_std)

    obs_partial = np.concatenate([np.concatenate(
        (np.zeros(impute - 1), [item])) for item in obs[impute:][0::impute]])
    obs_fill = np.concatenate([np.tile(item, impute)
                               for item in obs[impute:][0::impute]])
    obs_binary = [0.0 if item == 0 else 1.0 for item in obs_partial]

    count = 1
    time_till = np.zeros(len(obs_binary))
    for i in range(len(obs_binary)):
        if np.any(obs_binary[i] == np.array([1.])):
            count = 1
        else:
            time_till[i] = np.array([count])
            count += 1
    time_till_out = -(time_till - impute)

    f = open(dat_dir + "/dat/AR_obs_partial.txt", "w+")
    np.savetxt(f, obs_fill)
    f.close()

    f = open(dat_dir + "/dat/AR_obs_binary.txt", "w+")
    np.savetxt(f, obs_binary)
    f.close()

    f = open(dat_dir + "/dat/AR_time_till.txt", "w+")
    np.savetxt(f, time_till_out)
    f.close()


if __name__ == "__main__":
    data_gen(T=5000, impute=1, x0=10.0,
             theta=np.array([5.0, .5, 3.0]), obs_std=1.)
