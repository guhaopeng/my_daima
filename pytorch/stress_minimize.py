import os
import torch
import numpy as np  
import matplotlib.pyplot as plt  
from Stress_3D_Sensitivity_Comp import Stress_3D_Sensitivity_Comp  
from plot_von_Mises import plot_von_Mises


def stress_minimize(x, Hs, H, nelx, nely, nelz, show_plot=True, save_dir=None):
    pl = 3.0 
    q = 0.5 
    p = 10.0
    
    # x is a torch.Tensor with requires_grad=True
    x_vec = x.view(-1)
    
    # Filter: x_filtered = (H @ x_vec) / Hs
    if H.is_sparse:
        Hx = torch.sparse.mm(H, x_vec.unsqueeze(1)).squeeze(1)
    else:
        Hx = torch.matmul(H, x_vec)
    x_filtered = Hx / Hs.view(-1)
    
    # Physics forward pass (with PyTorch autograd tracking)
    pnorm, MISES = Stress_3D_Sensitivity_Comp(x_filtered, nelx, nely, nelz, pl, q, p)
    
    f0val = pnorm
    # Constraints
    fval = x_filtered.mean() - 0.3
    
    # Autograd: automatically compute gradients
    df0dx = torch.autograd.grad(f0val, x, retain_graph=True)[0].view(-1, 1)
    dfdx = torch.autograd.grad(fval, x, retain_graph=True)[0].view(1, -1)
    
    if show_plot or save_dir is not None:
        # Detach for plotting
        x_plot = x_filtered.detach().cpu().numpy().reshape((nely, nelx, nelz), order="F")
        MISES_np = MISES.detach().cpu().numpy()
        
        plt.figure(1)
        plt.clf()
        mat2d = np.flipud(x_plot[:, :, 0])    
        plt.contourf(mat2d, levels=[0.5, 1.5], colors=["black"])
        plt.gca().set_aspect("equal", adjustable="box")
        plt.axis("off")                           
        plt.title("material layout")              
        plt.gcf().set_facecolor("white")          
        plt.draw()                                
        MISES_plot = MISES_np * (0.5 * np.sign(x_plot.flatten(order="F") - 0.5) + 0.5)
        MISES_plot = MISES_plot.reshape((nely, nelx, nelz), order="F")

        plt.figure(2)                             
        plt.clf()                                 
        im = plt.imshow(MISES_plot[:, :, 0], cmap="jet", aspect="equal", origin="upper")
        plt.axis("off")                           
        plt.title("Von-Mises Stress")             
        plt.colorbar(im)                          
        plt.draw()                                
        if show_plot:
            plt.pause(0.001)                          
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.figure(1).savefig(os.path.join(save_dir, "material_layout.png"), dpi=300, bbox_inches="tight")
            plt.figure(2).savefig(os.path.join(save_dir, "von_mises_stress.png"), dpi=300, bbox_inches="tight")
        
        dmin = float(np.min(x_plot))
        dmax = float(np.max(x_plot))
        if dmin <= 0.5 <= dmax and min(x_plot.shape) >= 2:
            plt.figure(3)
            plot_von_Mises(x_plot, MISES_plot)
            if save_dir is not None:
                plt.figure(3).savefig(os.path.join(save_dir, "von_mises_3d.png"), dpi=300, bbox_inches="tight")

    return f0val.detach(), df0dx.detach(), fval.detach().view(1), dfdx.detach()
