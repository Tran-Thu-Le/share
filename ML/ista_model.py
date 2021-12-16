
import torch 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set_theme()

def dic(x, t, w):
    diff = x.reshape(-1, 1) - t.reshape(1, -1) 
    g = torch.pow(2, -diff**2/w**2)
    norms = g.norm(dim=0) 
    # assert norms.shape[0] == len(t)
    # print(norms.shape)
    return g/norms

def setup(x1, x2):
    m, n = 100, 16
    x = torch.linspace(0., 1., m, dtype=torch.float64)
    t = torch.linspace(0., 1., n, dtype=torch.float64) 
    w = 0.1
    atoms = dic(x, t, w) 

    k = 4
    ids = [n//k, (k-1)*(n//k)]
    coeffs = torch.zeros(n, dtype=torch.float64)
    coeffs[ids[0]] = x1
    coeffs[ids[1]] = x2
    coeffs = coeffs.reshape(-1, 1)
    y = atoms @ coeffs 

    infor = {
        "m": m,
        "n": n,
        "x": x,
        "t": t,
        "w": w,
        "ids": ids,
        "coeffs": coeffs,

    }

    lbd = 0.1

    return y, atoms, lbd, infor

def ISTA(y, atoms, lbd, max_iters=500):
    m, n = atoms.shape 
    assert y.shape == (m, 1) 
    assert y.dtype == atoms.dtype == torch.float64
    x = torch.zeros(n, dtype=torch.float64).reshape(-1, 1)
    A = atoms 
    _, s, _ = torch.linalg.svd(A)
    L = s.max()**2 
    x_list = []
    for i in range(max_iters):
        x = x + (1/L) * A.T @ (y- A @ x) 
        x = (x - lbd/L).clip(min=0.) 
        x_list += [x]
    return x_list

def model(plot=False):
    y, atoms, lbd, infor = setup(1, 0.9)
    x_list = ISTA(y, atoms, lbd, max_iters=200)

    # computing
    y_hat_list = [atoms @ x for x in x_list]
    y_hat = y_hat_list[-1]
    r_list = [y-yh for yh in y_hat_list]
    max_cert_list = [(atoms.T @ r).max() for r in r_list] 
    u_list = [(lbd/mc)*r for mc, r in zip(max_cert_list, r_list)]
    primal_list = [0.5* r.norm()**2 + lbd*x.norm(p=1) for r, x in zip(r_list, x_list)]
    dual_list = [0.5*y.norm()**2 - 0.5*(y-u).norm()**2 for u in u_list] 
    gap_list = [p-d for p, d in zip(primal_list, dual_list)]

    if plot==True:
        f, ax  = plt.subplots(1, 3, figsize=(21, 5))

        ax0 = ax[0]
        ax0.plot(infor["x"], y.reshape(-1), label="y")
        ax0.plot(infor["x"], y_hat.reshape(-1), label="y_hat")
        ax0.legend()

        ax1 = ax[1] 
        vals = [(cm-lbd)/lbd for cm in max_cert_list]
        ax1.plot(vals, label="(max_cert-lbd)/lbd")
        ax1.legend()
        ax1.set_yscale("log")
        
        ax2 = ax[2]
        ax2.plot(gap_list, label="dual gap")
        ax2.legend()
        ax2.set_yscale("log")
        plt.show() 

    # update infor 
    infor["x_list"] = x_list
    infor["y_hat_list"] = y_hat_list 
    infor["r_list"] = r_list 
    infor["max_cert_list"] = max_cert_list 
    infor["u_list"] = u_list 
    infor["primal_list"] = primal_list 
    infor["dual_list"] = dual_list 
    infor["gap_list"] = gap_list


    return y, atoms, lbd, infor

y, atoms, lbd, infor = model(plot=True)