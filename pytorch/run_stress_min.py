import os
import sys
import argparse
import warnings
import numpy as np
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(CURRENT_DIR)
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from prepare_filter import prepare_filter
from mma import asymp, gcmmasub, kktcheck
from stress_minimize import stress_minimize


import torch

def run_stress_min(
    nelx=100,
    nely=60,
    nelz=2,
    rmin=2.5,
    maxoutit=120,
    show_plot=True,
    suppress_linalg_warning=True,
):
    if suppress_linalg_warning:
        warnings.filterwarnings("ignore", message=".*Ill-conditioned matrix.*")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    
    x = 0.3 * torch.ones((nely, nelx, nelz), dtype=dtype, device=device)

    #3d-L shape initialization (Match L-shaped beam) ----------------
    # Remove top-right corner to make L-shape
    # Y (rows) goes from top to bottom. X (cols) goes from left to right.
    # Top-right is: rows 0 to nely//2, cols nelx//2 to nelx
    i_end = nely // 2
    j_start = nelx // 2
    x[:i_end, j_start:, :] = 1e-4 # -----------------------------------

    Hs, H_torch = prepare_filter(rmin, nelx, nely, nelz, device=device, dtype=dtype)
    Hs = Hs.view(-1, 1)

    m = 1
    epsimin = 1e-7
    n = x.numel()
    
    # In PyTorch, we can flatten and then reshape to emulate Fortran order by transposing
    # However, since x is 3D, x.reshape(-1, 1, order='F') in numpy corresponds to:
    # x.permute(2, 1, 0).reshape(-1, 1) in PyTorch if originally it was (nely, nelx, nelz)
    # Wait, numpy's 'F' order for (nely, nelx, nelz) is changing first index fastest.
    # PyTorch's transpose is better represented as permute(2, 1, 0).reshape(-1, 1) to match 'F'
    xval = x.permute(2, 1, 0).reshape(n, 1).clone()
    xval.requires_grad_(True)
    
    xold1 = xval.clone().detach()
    xold2 = xval.clone().detach()
    xlb = 1e-3 * torch.ones((n, 1), dtype=dtype, device=device)
    xub = torch.ones((n, 1), dtype=dtype, device=device)
    xmin = xlb.clone()
    xmax = xub.clone()
    low = xlb.clone()
    upp = xub.clone()
    c = torch.tensor([[1e4]], dtype=dtype, device=device)
    d = torch.tensor([[0.0]], dtype=dtype, device=device)
    a0 = 0.0
    a = torch.tensor([[0.0]], dtype=dtype, device=device)
    raa0 = 1e-4
    raa = 1e-4 * torch.ones((m, 1), dtype=dtype, device=device)
    raa0eps = 1e-7
    raaeps = 1e-7
    outeriter = 0
    kkttol = 0.0
    x_his = torch.zeros((n, maxoutit), dtype=dtype, device=device)
    innerit = 0
    result_dir = os.path.join(PYTHON_DIR, "result")

    if outeriter < 0.5:
        t0 = time.time()
        f0val, df0dx, fval, dfdx = stress_minimize(xval, Hs, H_torch, nelx, nely, nelz, show_plot=show_plot)
        t1 = time.time()
        print(f" Initial FEA time: {t1 - t0:.3f} s")
        df0dx = df0dx.view(n, 1)
        fval = fval.view(m, 1)
        dfdx = dfdx.view(m, n)
        outvector1 = torch.cat((torch.tensor([outeriter, innerit], dtype=dtype, device=device), xval.detach().view(-1)))
        outvector2 = torch.cat((torch.tensor([f0val.item()], dtype=dtype, device=device), fval.detach().view(-1)))
    else:
        outvector1 = None
        outvector2 = None

    kktnorm = kkttol + 1.0
    outit = 0
    last_residumax = None
    iter_log_lines = []

    while outit < maxoutit:
        outit += 1
        outeriter += 1
        
        t0_mma = time.time()
        low, upp, raa0, raa = asymp(
            outeriter,
            n,
            xval,
            xold1,
            xold2,
            xmin,
            xmax,
            low,
            upp,
            raa0,
            raa,
            raa0eps,
            raaeps,
            df0dx,
            dfdx,
        )
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, f0app, fapp = gcmmasub(
            m,
            n,
            outeriter,
            epsimin,
            xval,
            xmin,
            xmax,
            low,
            upp,
            raa0,
            raa,
            f0val,
            df0dx,
            fval,
            dfdx,
            a0,
            a,
            c,
            d,
        )
        t1_mma = time.time()
        mma_time = t1_mma - t0_mma
        
        xold2 = xold1.clone()
        xold1 = xval.clone().detach()
        xval = xmma.clone().detach().requires_grad_(True)
        save_dir = result_dir if outit == maxoutit else None
        
        t0_fea = time.time()
        f0val, df0dx, fval, dfdx = stress_minimize(
            xval,
            Hs,
            H_torch,
            nelx,
            nely,
            nelz,
            show_plot=show_plot,
            save_dir=save_dir,
        )
        t1_fea = time.time()
        fea_time = t1_fea - t0_fea
        
        df0dx = df0dx.view(n, 1)
        fval = fval.view(m, 1)
        dfdx = dfdx.view(m, n)
        iter_line = f" It.:{outit:5d}      P-norm Stress.:{f0val.item():11.4f}   Vol.:{xval.mean().item():7.3f}   FEA time: {fea_time:.3f} s   MMA time: {mma_time:.3f} s"
        print(iter_line)
        iter_log_lines.append(iter_line)
        residu, kktnorm, residumax = kktcheck(
            m,
            n,
            xmma,
            ymma,
            zmma,
            lam,
            xsi,
            eta,
            mu,
            zet,
            s,
            xmin,
            xmax,
            df0dx,
            fval,
            dfdx,
            a0,
            a,
            c,
            d,
        )
        outvector1 = torch.cat((torch.tensor([outeriter, innerit], dtype=dtype, device=device), xval.detach().view(-1)))
        outvector2 = torch.cat((torch.tensor([f0val.item()], dtype=dtype, device=device), fval.detach().view(-1)))
        x_his[:, outit - 1] = xmma.detach().view(-1)
        last_residumax = float(residumax)

    os.makedirs(result_dir, exist_ok=True)
    iteration_log_path = os.path.join(result_dir, "iteration_log.txt")
    with open(iteration_log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(iter_log_lines))

    return {
        "xval": xval.detach().cpu().numpy(),
        "f0val": f0val.item() if isinstance(f0val, torch.Tensor) else f0val,
        "fval": fval.detach().cpu().numpy(),
        "df0dx": df0dx.detach().cpu().numpy(),
        "dfdx": dfdx.detach().cpu().numpy(),
        "kktnorm": float(kktnorm),
        "residumax": last_residumax,
        "x_his": x_his.detach().cpu().numpy(),
        "outvector1": outvector1.detach().cpu().numpy() if outvector1 is not None else None,
        "outvector2": outvector2.detach().cpu().numpy() if outvector2 is not None else None,
        "result_dir": result_dir,
        "iteration_log_path": iteration_log_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nelx", type=int, default=60)
    parser.add_argument("--nely", type=int, default=60)
    parser.add_argument("--nelz", type=int, default=10)
    parser.add_argument("--rmin", type=float, default=2.5)
    parser.add_argument("--maxoutit", type=int, default=120)
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--show-linalg-warning", action="store_true")
    args = parser.parse_args()
    show_plot = True if args.show_plot else not args.no_plot
    run_stress_min(
        nelx=args.nelx,
        nely=args.nely,
        nelz=args.nelz,
        rmin=args.rmin,
        maxoutit=args.maxoutit,
        show_plot=show_plot,
        suppress_linalg_warning=not args.show_linalg_warning,
    )
