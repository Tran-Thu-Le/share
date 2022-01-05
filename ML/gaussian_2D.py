
import torch 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_theme()

def dic(w, t, sigma=0.05):
    """
        discritization of 2D gaussian
        2^{-||w-t||^2/sigma^2}
    """
    diff = w.T.view(-1, 2, 1) - t
    # print(diff.shape, "olala")
    dist = torch.norm(diff, dim=1)
    gauss = torch.pow(2, -dist**2/sigma**2)
    norm = torch.norm(gauss, dim=0)
    # print("heyhey", gauss.shape, norm.shape)
    assert norm.shape[0] == gauss.shape[1]
    return gauss/norm 

def set_up(lbd=0.1):
    sigma = 0.08

    wa, wb = -1., 2.
    wm = 30
    dw = (wb-wa)/wm
    m = wm**2
    w_axis = torch.linspace(wa, wb, wm+1, dtype=torch.float64) 
    w_axis = w_axis[: -1] + dw/2 
    w_grid = torch.meshgrid(w_axis, w_axis) 
    w = torch.cat([w_grid[0].reshape(1, -1), w_grid[1].reshape(1, -1)], 0) 
    #
    ta, tb = 0., 1.
    tn = 20
    dt = (tb-ta)/tn 
    n = tn**2
    t_axis = torch.linspace(ta, tb, tn+1, dtype=torch.float64) 
    t_axis = t_axis[: -1] + dt/2 
    t_grid = torch.meshgrid(t_axis, t_axis) 
    t = torch.cat([t_grid[0].reshape(1, -1), t_grid[1].reshape(1, -1)], 0) 



    atoms = dic(w, t, sigma=sigma) 


    id_true = [3+tn*3, 17+tn*7, 9+tn*17]
    t_true = t[:, id_true]
    atoms_true = atoms[:, id_true]
    c1=1.
    c2=0.9
    c3=0.9
    c_true = torch.tensor([c1, c2, c3], dtype=torch.float64).reshape(-1, 1)

    # id_true = None
    # t_true = torch.tensor([[0.2, 0.3, 0.8], [0.1, 0.9, 0.5]], dtype=torch.float64)
    # atoms_true = dic(w,t_true, sigma=sigma) 
    # c1=1.
    # c2=0.9
    # c3=0.9
    # c_true = torch.tensor([c1, c2, c3], dtype=torch.float64).reshape(-1, 1)


    y =  atoms_true @ c_true
    # lbd = 0.1

    # Lipchizt 
    id_mid = n//2 
    a = (atoms[:, [id_mid]] - atoms[:, [id_mid+1]]).norm()
    b = (t[:, [id_mid]] - t[:, [id_mid+1]]).norm()
    L = a/b 

    cert0 = (atoms.T @ y).reshape(-1)
    lbd_max = cert0.max()
    y_scale = (lbd/lbd_max)*y
    origin_m = torch.zeros(m, dtype=torch.float64).reshape(-1, 1)
    origin_n = torch.zeros(n, dtype=torch.float64).reshape(-1, 1)
    zero = torch.tensor(0., dtype=torch.float64)
    one = torch.tensor(1., dtype=torch.float64)
    Id = torch.eye(m, dtype=torch.float64) 
    

    infor = {
        "wa": wa,
        "wb": wb,
        "wm": wm,
        "m": m,
        "dw": dw,
        "w_axis": w_axis,
        "w_grid": w_grid,
        "w": w,
        
        "ta": ta,
        "tb": tb,
        "tn": tn,
        "n": n,
        "dt": dt,
        "t_axis": t_axis,
        "t_grid": t_grid,
        "t": t,

        "y": y,
        "atoms": atoms,
        "lbd": lbd,

        "y_norm": y.norm(),
        "sigma": sigma,
        "Lipchitz": L,
        "cert0": cert0,
        "lbd_max": lbd_max,
        "y_scale": y_scale,
        "t_true": t_true,
        "id_true": id_true,
        "c_true": c_true,
        "c1": c1, 
        "c2": c2,
        "origin_m": origin_m,
        "origin_n": origin_n,
        "zero": zero,
        "one": one,
        "Id": Id,
    }

    return infor 


def update_data(setup, data_pre=None):
    origin_m = setup["origin_m"]
    lbd = setup["lbd"]
    y = setup["y"]
    y_norm = setup["y_norm"]
    atoms=setup["atoms"] 
    t_points = setup["t"]


    # primal data
    if data_pre is None:
        iter = 0 
        t = None # position
        x = None # coefficients 
        x_TV = 0. 
        v = origin_m # Image of zero measure
    else:
        iter = data_pre["iter"][-1] + 1 
        t = data_pre["maxer"][-1]
        x = data_pre["cert_max"][-1] - lbd 
        x_TV = data_pre["x_TV"][-1] + x 
        v = data_pre["v"][-1] + x*data_pre["a"][-1] 

    r = y-v 
    p_val = 0.5*r.norm()**2 + lbd*x_TV 

    # cert data
    cert = (atoms.T @ r).reshape(-1) 
    id = cert.argmax()
    maxer = t_points[:, [id]]
    cert_max = cert[id] 
    a = atoms[:, [id]] 

    # dual data
    u = (lbd/cert_max)*r 
    d_val = 0.5*y_norm**2 - 0.5*(y-u).norm()**2
    if iter==0:
        pass 
    else:
        if d_val > data_pre["d_val"][-1]:
            pass
        else:
            u = data_pre["u"][-1]
            d_val = data_pre["d_val"][-1]

    gap = p_val - d_val 
    R = (2*gap.abs()).sqrt() 


    # update data
    keys = ["iter", "t", "x", "x_TV", "v", "r", "p_val",
            "cert", "id", "maxer", "cert_max", "a",
            "u", "d_val", "gap", "R"] 

    vals = [iter, t, x, x_TV, v, r, p_val,
            cert, id, maxer, cert_max, a,
            u, d_val, gap, R]

    if iter==0:
        data = {}
        for k, v in zip(keys, vals):
            data[k] = [v] 
        return data 
    else:
        for k, v in zip(keys, vals):
            data_pre[k] += [v] 
        return data_pre


def solve(setup):
    data=None
    for i in range(setup["t_true"].shape[1]+1):
        data = update_data(setup, data_pre=data) 
    return data


def plot(setup, data):

    f, ax = plt.subplots( figsize=(7*1, 5*1))
    color = "Reds"
    levels=20
    vmin=0. 
    vmax = 1.
    color2="black"

    # inputs
    t_grid = setup["t_grid"]
    ta, tb = setup["ta"], setup["tb"]
    tn = setup["tn"] 
    cert0 = setup["cert0"]
    t_true = setup["t_true"] 
    # t0, t1 = t_true[:, [0]], t_true[:, [1]]
    w = setup["w"] 
    sigma = setup["sigma"]
    lbd=setup["lbd"] 
    y=setup["y"]
    y_scale=setup["y_scale"] 
    c0, c1, c2 = data["u"][:3]
    R0, R1, R2 = data["R"][:3]
    _, t1, t2, t3 = data["t"]


    ax=ax
    ax.contourf(t_grid[0], t_grid[1], cert0.reshape(tn, tn),
                cmap=color, levels=levels, vmin=vmin, vmax=vmax)
    ax.scatter(t1[0], t1[1], color=color2, marker="x")
    ax.scatter(t2[0], t2[1], color=color2, marker="x")
    ax.scatter(t3[0], t3[1], color=color2, marker="x")
    ax.set_title(f"Initial cert <y, a_t> and locations of estimate spikes")
    ax.set_xlim(ta, tb)
    ax.set_ylim(ta, tb) 


def get_demo():
    text = """
    DEMO CODE:\n 
    import torch \n
    import matplotlib.pyplot as plt \n
    import seaborn as sns \n
    sns.set_theme() \n 
    from gaussian_2D import set_up, solve, plot, get_demo \n
    \n 
    # setup = set_up() \n
    # data = solve(setup) \n
    # plot(setup, data)\n
    get_demo()
    \n
    DEMO RESULT:\n
    """
    print(text)

    setup = set_up() 
    data = solve(setup) 
    plot(setup, data)
