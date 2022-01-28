
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_theme()
import math 

def fista_scaling_factor(y, A, lbd, max_iters=1000):
    m, n = A.shape 
    zero = np.zeros(n).reshape(-1, 1) 
    x = zero
    eigs = np.linalg.eigvals(A.T @ A)
    L = eigs.real.max()
    

    # print(y.shape, A.shape, x.shape, L, lbd)

    x_list = []
    beta = 1.
    for i in range(max_iters):
        x_copy = x

        # ista
        x = x + (1/L)*A.T @ ( y-A@x) 
        x = np.clip(x - lbd/L, a_min=0, a_max=None)

        # fista
        beta_copy = beta 
        beta = (1. + math.sqrt(1. + 4. * (beta**2))) / 2.
        x = x + ((beta_copy - 1.) / beta) * (x - x_copy)

        # scaling factor
        v = A@ x 
        a = (y*v).sum() - lbd * x.sum()
        b = (v**2).sum()
        scaling_factor = a/b 
        x = scaling_factor * x

        x_list += [x]

    return x_list

def get_data(y, A, lbd, x_list):

    data = {
        "x": [],
        "r": [],
        "lbd_r": [],
        "u": [],
        "p_val": [],
        "d_val": [],
        "gap": [],
    }

    for x in x_list:
        r = y-A @ x
        lbd_r = (A.T @ r).max()
        u = (lbd/lbd_r)*r
        p_val = 0.5 * np.linalg.norm(r)**2 + lbd * x.sum()
        d_val = 0.5 * np.linalg.norm(y)**2 - 0.5 * np.linalg.norm(y-u)**2 
        gap = p_val - d_val 

        data["x"] += [x]
        data["r"] += [r] 
        data["lbd_r"] += [lbd_r]
        data["u"] += [u]
        data["p_val"] += [p_val]
        data["d_val"] += [d_val]
        data["gap"] += [gap]

    return data

def dic(w, t, sigma):
    diff = w.reshape(-1, 1) - t.reshape(1, -1) 
    gausses = np.power(2, -diff**2/sigma**2)
    norms = np.linalg.norm(gausses, axis=0)
    return gausses/norms 

def demo():
    np.random.seed(123)
    m, n = 100, 200 
    w = np.linspace(0., 1., m)
    t = np.linspace(0., 1., n) 
    sigma=0.05
    A = dic(w, t, sigma)
    # norms = np.linalg.norm(A, axis=0)
    # A = A/norms 

    # print((A[:, 1]**2).sum())

    ids = [50, 100, 160]
    coeffs = np.array([1., 1., 1.]).reshape(-1, 1)
    y = A[:, ids] @ coeffs 

    lbd_max = (A.T @ y).max()
    lbd = 0.5*lbd_max 

    

    x_list = fista_scaling_factor(y, A, lbd, max_iters=10000)
    data = get_data(y, A, lbd, x_list)

    f, axs = plt.subplots(1, 3, figsize=(7*3, 5))

    ax = axs[0]
    ax.plot(data["gap"], label="p_val")
    ax.set_yscale("log")
    ax.set_title("Dual gap")
    ax.set_xlabel("Iters")

    ax = axs[1]
    max_cert = [1-lbd/lbd_r for lbd_r in data["lbd_r"]]
    ax.plot(max_cert, label="1-lbd/lbd_r")
    ax.set_yscale("log")
    ax.set_title("Max cert")
    ax.set_xlabel("Iters")

    ax = axs[2]
    ax.plot(w, y.reshape(-1), label="y")
    x_last = x_list[-1]
    ax.plot(t, x_last.reshape(-1), label="x")
    ax.legend()
    ax.set_title("y and x")
    # ax.set_xlabel("Iters")

    # ax.plot(x_list[-1])
    
    plt.show()

# demo()