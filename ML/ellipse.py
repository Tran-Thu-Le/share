

class Ellipse:

    def __init__(self, center, matrix):
        m = matrix.shape[0]
        assert matrix.shape==(m, m) and center.shape ==(m, 1)
        assert matrix.dtype==torch.float64 and center.dtype == torch.float64
        self.center = center 
        self.matrix = matrix 
        eigs, vecs = torch.linalg.eig(self.matrix)
        self.eigs = eigs.real
        self.vecs = vecs.real
        self.dim = m
    
    # ATTRIBUTES
    def mat_sqrt(self):
        return self.vecs @ torch.diag(self.eigs.sqrt()) @ self.vecs.T

    def mat_inv(self):
        return self.vecs @ torch.diag(1/self.eigs) @ self.vecs.T

    def radius(self):
        return self.eigs.max().sqrt()

    def volume(self):
        logdet = torch.logdet(self.matrix)
        m = self.matrix.shape[0]
        log_beta_n = .5 * m * torch.log(torch.tensor(pi)) - torch.lgamma(torch.tensor(.5 * m + 1.)) 
        out = torch.exp( log_beta_n + .5 * logdet )
        return out

    def balance(self):
        return self.eigs.max().sqrt() - self.eigs.min().sqrt()

    def copy(self):
        return Ellipse(self.center, self.matrix)


    # supporting tools

    def _convex_combination(self, ell2, lamb):
        # ell1 = self
        # find lamb maximize det(E) 
        E1, E2 = self.mat_inv(), ell2.mat_inv()
        x1, x2 = self.center, ell2.center
        X = lamb*E1 + (1-lamb)*E2 
        try:
            X_inv = torch.inverse(X)
        except:
            raise Exception("convex_combination line ...: Check the matrix of ell1 and ell2")
        k = 1. - lamb*(1-lamb)* ((x2-x1).T @ E2 @ X_inv @ E1 @ (x2-x1))
        # print(f'110: k={k}')
        x0 = X_inv @ (lamb * E1 @ x1 + (1-lamb) * E2 @ x2)
        E = X/k
        E_inv = torch.inverse(E)
        return Ellipse(x0, E_inv)

    def _get_point(self, p):
        """
            get the intersection point of ellipse
            and segment joining p and ellipse's center.
        """
        assert p.shape==(self.dim, 1) 
        g = p - self.center 
        vec = g/ (g * (self.mat_inv() @ g)).sum().sqrt()
        return self.center + vec 

    def _get_tangent_plane(self, p):
        """
            p must be a tangent point
        """
        assert p.shape==(self.dim, 1)
        g = self.mat_inv() @ (p-self.center)
        level = 1. + (g*self.center).sum()
        return g, level

    def contains_point(self, p):
        m = self.dim 
        assert p.shape == (m, 1)
        a = (p * (self.mat_inv() @ p)).sum() 
        if a <=1.+1e-9:
            return a-1
        else:
            return a-1

    # Basic cuts
    def one_plane_cut(self, g, lamb, get_infor=True):
        c = self.center 
        mat = self.matrix 

        m = mat.shape[0]
        g_norm_2 = (g * (mat @ g)).sum()
        g_norm = g_norm_2.sqrt()
        h =  (g.T @ c).squeeze()-lamb
        alpha = h/g_norm 
        # print(f"ellipse 14: h={h} g_norm^2={g_norm_2} g_norm={g_norm} alpha={alpha}") # lbd
        if alpha >1:
            raise Exception("WARNING: Ellipse.new_ellipse line 47: Intersection of ellipsoid and half-space is empty")
        elif -1/m<=alpha<=1.:
        # if alpha.abs()<=1.:
            g_bar = g/g_norm 
            delta_1 = (1+alpha*m)/(m+1)
            delta_2 = (m**2*(1-alpha**2))/(m**2-1)
            delta_3 = (2*(1+alpha*m))/((m+1)*(1+alpha))
            c2 = c - delta_1 * mat @ g_bar
            g_mat = g_bar @ g_bar.T
            mat2 = delta_2*(mat - delta_3 * mat @ g_mat @ mat)
            ell =  Ellipse(c2, mat2)
            ell.alpha = alpha
            infor = {
                "alpha": alpha,
                "change": True,
                'complexity': m**2,
            }
            # return ell, infor
        else:
            infor = {
                "alpha": alpha,
                "change": False,
                'complexity': m**2,
            }
            ell =  self.copy()

        if get_infor == True:
            return ell, infor 
        else:
            return ell

    def radius_compression_cut(self, g, lamb, get_infor=True):
        c = self.center 
        mat = self.matrix 

        m = mat.shape[0]
        g_norm_2 = (g * (mat @ g)).sum()
        g_norm = g_norm_2.sqrt()
        h =  (g.T @ c).squeeze()-lamb
        alpha = h/g_norm 
        # print(f"130 Ellipse.radius_compression_cut : alpha={alpha}")
        if alpha > 1. +1e-6:
            raise Exception("WARNING: Ellipse.new_ellipse line 47: Intersection of ellipsoid and half-space is empty")
        elif 0.<alpha<=1.+1e-6:
            g_bar = g/g_norm 
            delta_1 = alpha
            delta_2 = 1-alpha**2
            delta_3 = (2*alpha)/(1+alpha)
            c2 = c - delta_1 * mat @ g_bar
            g_mat = g_bar @ g_bar.T
            mat2 = delta_2*(mat - delta_3 * mat @ g_mat @ mat)
            ell =  Ellipse(c2, mat2)
            ell.alpha = alpha
            infor = {
                "alpha": alpha,
                "change": True,
                'complexity': m**2,
            }
        else:
            infor = {
                "alpha": alpha,
                "change": False,
                'complexity': m**2,
            }
            ell =  self.copy()

        if get_infor == True:
            return ell, infor 
        else:
            return ell

    def parallel_cut(self, g, beta, beta_hat, get_infor=True):
        """
            plane: beta_hat <= <g, x> <= beta
        """
        n = self.matrix.shape[0]
        assert g.shape==(n, 1)
        c = self.center 
        P = self.matrix

        g_norm = (g * (P @ g)).sum().sqrt()
        g_bar = g/g_norm
        g_mat = P @ g_bar


        alpha = ((g*c).sum()-beta )/g_norm 
        alpha_hat = (beta_hat-(g*c).sum())/g_norm 

        assert alpha*alpha_hat < 1/n 
        assert alpha <= -alpha_hat <= 1. 

        rho = (4*(1-alpha**2)*(1-alpha_hat**2)+n**2*(alpha_hat**2-alpha**2)**2).sqrt()
        sigma = (1/(n+1))*(n+(2/(alpha-alpha_hat)**2)*(1-alpha*alpha_hat-rho/2))
        tau = (alpha-alpha_hat)*sigma/2 
        delta = (n**2/(n**2-1))*(1-(alpha**2+alpha_hat**2-rho/n)/2)

        c2 = c - tau * g_mat 
        P2 = delta*(P - sigma * (g_mat @ g_mat.T)) 

        infor = {
            "alpha": alpha,
            "alpha_hat": alpha_hat,
            "complexity": n**2,
        }
        if get_infor:
            return Ellipse(c2, P2), infor
        else: 
            return Ellipse(c2, P2)

    def fusion_cut(self, ell2, lamb_init=0.5):
        def func(ell2, lamb):
            ell = self.convex_combination(ell2, lamb)
            E = ell.mat_inv()
            return torch.det(E)
        lamb_list = torch.linspace(0., 1., 50)
        det_list = [func(ell2, lamb) for lamb in lamb_list]
        det_max= max(det_list)
        id_max = det_list.index(det_max)
        lamb_opt = lamb_list[id_max]
        # print(f"145 fusion: lamb_opt={lamb_opt}")
        ell = self._convex_combination(ell2, lamb_opt) 
        return ell


    # variants of one_plane_cut/radius_compression_cut              
    def random_cut(self, atoms, lamb):
        ell = self 
        n = atoms.shape[1]
        rand_perm = torch.randperm(n)
        for i in range(n):
            k = rand_perm[i] 
            ell, _ = ell.one_plane_cut(atoms[:, [k]], lamb)  
        return ell 

    def atomic_and_deep_cut(self, y, atoms, lbd, mode="one_plane_cut", get_infor=False):
        c = self.center
        mat = self.matrix 

        a = (c * atoms).sum(0) - lbd
        b = (atoms * (mat @ atoms) ).sum(dim=0).sqrt()
        alpha = a/b 
        
        if (a<0.).all():
            # print("---->")
            g = c - y 
            level = (g.T @ c).squeeze()
            if mode == "one_plane_cut":
                ell, infor = self.one_plane_cut(g, level) 
            else:
                ell, infor = self.radius_compression_cut(g, level)
        else:
            # print("====>")
            id_max = alpha.argmax() 
            g = atoms[:, [id_max]] 
            if mode == "one_plane_cut":
                ell, infor = self.one_plane_cut(g, lbd) 
            else:
                ell, infor = self.radius_compression_cut(g, lbd)

        # print(f"ellipse.atomic_and_deep_cut 249: alpha={infor['alpha']}")
        if get_infor == True:
            return ell, infor 
        else:
            return ell

    def tangent_cut(self, ell):
        assert self.dim == ell.dim 
        point = ell._get_point(self.center)
        g, level = ell._get_tangent_plane(point)
        ell_new, infor = self.one_plane_cut(g, level)
        return ell_new

    def fusion(self, ell2, lamb_init = 0.5, get_infor=False):
        # ell1 = self

        def func(ell2, lamb):
            ell = self.convex_combination(ell2, lamb)
            E = ell.mat_inv()
            return torch.det(E)
        lamb_list = torch.linspace(0., 1., 50)
        det_list = [func(ell2, lamb) for lamb in lamb_list]
        det_max= max(det_list)
        id_max = det_list.index(det_max)
        lamb_opt = lamb_list[id_max]
        # print(f"145 fusion: lamb_opt={lamb_opt}")
        ell = self.convex_combination(ell2, lamb_opt) 

        infor = {
            "lamb": lamb_opt,
        }
        if get_infor==True:
            return ell, infor 
        else:
            return ell

    def eigenvector_correlation_cut(self, atoms, lbd, mode="one_plane_cut"):
        V = self.vecs 
        A = atoms 
        n, N = V.shape[1], A.shape[1]
        cosine = A.T @ V 
        norms = A.norm(dim=0).reshape(-1, 1) * V.norm(dim=0).reshape(1, -1)
        assert cosine.shape == norms.shape 
        cosine = cosine/norms
        max_column_data = cosine.max(dim=0) 
        for i in range(n):
            k = max_column_data.indices[i]
            a = A[:, [k]]
            if mode == "one_plane_cut":
                ell, infor = self.one_plane_cut(a, lbd) 
            else:
                ell, infor = self.radius_compression_cut(a, lbd ) 

            print(f"Ellipse 293: alpha={infor['alpha']}")
        return ell

    def eigenvector_perpendicular_cut(self, ell2):
        """
            cut self by parallel cut using self.eigenvactor_max as normal vector
        """
        id_max = self.eigs.argmax()  
        v = self.vecs[:, [id_max]] 
        v_scale = v/ (v * (ell2.mat_inv() @ v)).sum().sqrt() 
        g = self.matrix @ v_scale 
        level = 1. + (g * self.center).sum()
        level_hat = -1. + (g * self.center).sum() 

        ell, infor = self.parallel_cut(g, level, level_hat, get_infor=True)

        return ell



    # 2D PLOTTING TOOLs
    def plot_tangent_plane(self, p, ax, xmin, xmax, **kwargs):
        """
            p must be a tangent point
        """
        assert self.dim==2
        g, level = self.get_tangent_plane(p)
        a = (level-g[0][0]*xmin)/g[1][0]
        b = (level-g[0][0]*xmax)/g[1][0]
        ax.plot([xmin, xmax], [a, b], **kwargs)

    def plot(self, ax, **kwargs):
        assert self.dim==2
        t = 2*pi*torch.linspace(0., 1., 100, dtype = torch.float64)
        x, y = torch.cos(t), torch.sin(t)
        x2, y2 = self.center + self.mat_sqrt() @ torch.cat([x, y]).reshape(2, -1)
        ax.plot(x2, y2, **kwargs)