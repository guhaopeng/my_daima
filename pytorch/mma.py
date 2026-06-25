# SPDX-License-Identifier: GPL-3.0-or-later

"""
GCMMA-MMA-Python

This file is part of GCMMA-MMA-Python. GCMMA-MMA-Python is licensed under the terms of GNU 
General Public License as published by the Free Software Foundation. For more information and 
the LICENSE file, see <https://github.com/arjendeetman/GCMMA-MMA-Python>. 

The orginal work is written by Krister Svanberg in MATLAB. This is the Python implementation 
of the code written by Arjen Deetman.

Functionality:
- `mmasub`: Solves the MMA subproblem.
- `gcmmasub`: Solves the GCMMA subproblem.
- `subsolv`: Performs a primal-dual Newton method to solve subproblems.
- `kktcheck`: Checks the Karush-Kuhn-Tucker (KKT) conditions for the solution.

Dependencies:
- numpy: Numerical operations and array handling.
- scipy: Sparse matrix operations and linear algebra.

To use this module, import the desired functions and provide the necessary arguments 
according to the specific problem being solved.
"""

# Loading modules
from __future__ import division
import torch
from typing import Tuple

def mmasub(m: int, n: int, iter: int, xval: torch.Tensor, xmin: torch.Tensor, xmax: torch.Tensor,
           xold1: torch.Tensor, xold2: torch.Tensor, f0val: torch.Tensor, df0dx: torch.Tensor, fval: torch.Tensor, dfdx: torch.Tensor,
           low: torch.Tensor, upp: torch.Tensor, a0: float, a: torch.Tensor, c: torch.Tensor,
           d: torch.Tensor, move: float = 0.5, asyinit: float = 0.5, asydecr: float = 0.7, asyincr: float = 1.2, 
           asymin: float = 0.01, asymax: float = 10, raa0: float = 0.00001, 
           albefa: float = 0.1, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor, 
                                        torch.Tensor, float, torch.Tensor, torch.Tensor]:

    """
    Solve the MMA (Method of Moving Asymptotes) subproblem for optimization.
    """
    device = xval.device
    dtype = xval.dtype

    df0dx = df0dx.view(n, 1)
    dfdx = dfdx.view(m, n)

    epsimin = 0.0000001
    eeen = torch.ones((n, 1), dtype=dtype, device=device)
    eeem = torch.ones((m, 1), dtype=dtype, device=device)
    zeron = torch.zeros((n, 1), dtype=dtype, device=device)
    
    # Calculation of the asymptotes low and upp
    if iter <= 2:
        low = xval - asyinit * (xmax - xmin)
        upp = xval + asyinit * (xmax - xmin)
    else:
        zzz = (xval - xold1) * (xold1 - xold2)
        factor = eeen.clone()
        factor[zzz > 0] = asyincr
        factor[zzz < 0] = asydecr
        low = xval - factor * (xold1 - low)
        upp = xval + factor * (upp - xold1)
        lowmin = xval - asymax * (xmax - xmin)
        lowmax = xval - asymin * (xmax - xmin)
        uppmin = xval + asymin * (xmax - xmin)
        uppmax = xval + asymax * (xmax - xmin)
        low = torch.maximum(low, lowmin)
        low = torch.minimum(low, lowmax)
        upp = torch.minimum(upp, uppmax)
        upp = torch.maximum(upp, uppmin)

    # Calculation of the bounds alfa and beta
    zzz1 = low + albefa * (xval - low)
    zzz2 = xval - move * (xmax - xmin)
    zzz = torch.maximum(zzz1, zzz2)
    alfa = torch.maximum(zzz, xmin)
    zzz1 = upp - albefa * (upp - xval)
    zzz2 = xval + move * (xmax - xmin)
    zzz = torch.minimum(zzz1, zzz2)
    beta = torch.minimum(zzz, xmax)

    # Calculations of p0, q0, P, Q and b
    xmami = xmax - xmin
    xmami_eps = 0.00001 * eeen
    xmami = torch.maximum(xmami, xmami_eps)
    xmami_inv = eeen / xmami
    ux1 = upp - xval
    ux2 = ux1 * ux1
    xl1 = xval - low
    xl2 = xl1 * xl1
    ux_inv = eeen / ux1
    xl_inv = eeen / xl1
    
    p0 = torch.maximum(df0dx, zeron)
    q0 = torch.maximum(-df0dx, zeron)
    pq0 = 0.001 * (p0 + q0) + raa0 * xmami_inv
    p0 = p0 + pq0
    q0 = q0 + pq0
    p0 = p0 * ux2
    q0 = q0 * xl2
    
    P = torch.maximum(dfdx, torch.zeros_like(dfdx))
    Q = torch.maximum(-dfdx, torch.zeros_like(dfdx))
    PQ = 0.001 * (P + Q) + raa0 * torch.matmul(eeem, xmami_inv.T)
    P = P + PQ
    Q = Q + PQ
    P = P * ux2.view(1, -1)
    Q = Q * xl2.view(1, -1)
    b = torch.matmul(P, ux_inv) + torch.matmul(Q, xl_inv) - fval

    # Solving the subproblem using the primal-dual Newton method
    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = subsolv(
        m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d)
    
    # Return values
    return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp

def gcmmasub(m: int, n: int, iter: int, epsimin: float, xval: torch.Tensor, xmin: torch.Tensor, 
             xmax: torch.Tensor, low: torch.Tensor, upp: torch.Tensor, raa0: float, raa: torch.Tensor, 
             f0val: torch.Tensor, df0dx: torch.Tensor, fval: torch.Tensor, dfdx: torch.Tensor, a0: float, 
             a: torch.Tensor, c: torch.Tensor, d: torch.Tensor, albefa: float = 0.01, move: float = 0.2, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, 
            float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    
    """
    Solve the GCMMA (Generalized Convex Method of Moving Asymptotes) subproblem for optimization.
    """
    
    device = xval.device
    dtype = xval.dtype

    df0dx = df0dx.view(n, 1)
    dfdx = dfdx.view(m, n)

    eeen = torch.ones((n, 1), dtype=dtype, device=device)
    zeron = torch.zeros((n, 1), dtype=dtype, device=device)


    # Calculations of the bounds alfa and beta
    zzz1 = low + albefa * (xval - low)
    zzz2 = xval - move * (xmax - xmin)
    zzz = torch.maximum(zzz1, zzz2)
    alfa = torch.maximum(zzz, xmin)
    zzz1 = upp - albefa * (upp - xval)
    zzz2 = xval + move*(xmax-xmin)
    zzz = torch.minimum(zzz1, zzz2)
    beta = torch.minimum(zzz, xmax)

    # Calculations of p0, q0, r0, P, Q, r and b.
    xmami = xmax - xmin
    xmami_eps = 0.00001 * eeen
    xmami = torch.maximum(xmami, xmami_eps)
    xmami_inv = eeen / xmami
    ux1 = upp - xval
    ux2 = ux1 * ux1
    xl1 = xval - low
    xl2 = xl1 * xl1
    ux_inv = eeen / ux1
    xl_inv = eeen / xl1

    # Initializations for p0, q0
    p0 = torch.maximum(df0dx, zeron)
    q0 = torch.maximum(-df0dx, zeron)
    pq0 = p0 + q0
    p0 = p0 + 0.001 * pq0
    q0 = q0 + 0.001 * pq0
    p0 = p0 + raa0 * xmami_inv
    q0 = q0 + raa0 * xmami_inv
    p0 = p0 * ux2
    q0 = q0 * xl2
    r0 = f0val - torch.matmul(p0.T, ux_inv) - torch.matmul(q0.T, xl_inv)
    
    P = torch.maximum(dfdx, torch.zeros_like(dfdx))
    Q = torch.maximum(-dfdx, torch.zeros_like(dfdx))
    PQ = P + Q
    P = P + 0.001 * PQ
    Q = Q + 0.001 * PQ
    P = P + torch.matmul(raa, xmami_inv.T)
    Q = Q + torch.matmul(raa, xmami_inv.T)
    P = P * ux2.view(1, -1)
    Q = Q * xl2.view(1, -1)
    r = fval - torch.matmul(P, ux_inv) - torch.matmul(Q, xl_inv)
    b = -r

    # Solving the subproblem using the primal-dual Newton method
    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d)

    # Calculations of f0app and fapp
    ux1 = upp - xmma
    xl1 = xmma - low
    ux_inv = eeen / ux1
    xl_inv = eeen / xl1
    f0app = r0 + torch.matmul(p0.T, ux_inv) + torch.matmul(q0.T, xl_inv)
    fapp = r + torch.matmul(P, ux_inv) + torch.matmul(Q, xl_inv)

    # Return values
    return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, f0app, fapp

def subsolv(m: int, n: int, epsimin: float, low: torch.Tensor, upp: torch.Tensor, alfa: torch.Tensor, 
            beta: torch.Tensor, p0: torch.Tensor, q0: torch.Tensor, P: torch.Tensor, Q: torch.Tensor, 
            a0: float, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, 
            torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor]:
    
    """
    Solve the MMA (Method of Moving Asymptotes) subproblem for optimization.
    """

    device = low.device
    dtype = low.dtype

    een = torch.ones((n, 1), dtype=dtype, device=device)
    eem = torch.ones((m, 1), dtype=dtype, device=device)
    epsi = 1.0
    epsvecn = epsi * een
    epsvecm = epsi * eem
    x = 0.5 * (alfa + beta)
    y = eem.clone()
    z = torch.tensor([[1.0]], dtype=dtype, device=device)
    lam = eem.clone()
    xsi = een / (x - alfa)
    xsi = torch.maximum(xsi, een)
    eta = een / (beta - x)
    eta = torch.maximum(eta, een)
    mu = torch.maximum(eem, 0.5 * c)
    zet = torch.tensor([[1.0]], dtype=dtype, device=device)
    s = eem.clone()
    itera = 0

    # Start while loop for numerical stability
    while epsi > epsimin:
        epsvecn = epsi * een 
        epsvecm = epsi * eem
        ux1 = upp - x
        xl1 = x - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        uxinv1 = een / ux1
        xlinv1 = een / xl1
        plam = p0 + torch.matmul(P.T, lam)
        qlam = q0 + torch.matmul(Q.T, lam)
        gvec = torch.matmul(P, uxinv1) + torch.matmul(Q, xlinv1)
        dpsidx = plam / ux2 - qlam / xl2
        rex = dpsidx - xsi + eta
        rey = c + d * y - mu - lam
        rez = a0 - zet - torch.matmul(a.T, lam)
        relam = gvec - a * z - y + s - b
        rexsi = xsi * (x - alfa) - epsvecn
        reeta = eta * (beta - x) - epsvecn
        remu = mu * y - epsvecm
        rezet = zet * z - epsi
        res = lam * s - epsvecm
        residu1 = torch.cat((rex, rey, rez), dim=0)
        residu2 = torch.cat((relam, rexsi, reeta, remu, rezet, res), dim=0)
        residu = torch.cat((residu1, residu2), dim=0)
        residunorm = torch.sqrt(torch.matmul(residu.T, residu)).item()
        residumax = torch.max(torch.abs(residu)).item()
        ittt = 0

        # Start inner while loop for optimization
        while (residumax > 0.9 * epsi) and (ittt < 200):
            ittt += 1
            itera += 1
            ux1 = upp - x
            xl1 = x - low
            ux2 = ux1 * ux1
            xl2 = xl1 * xl1
            ux3 = ux1 * ux2
            xl3 = xl1 * xl2
            uxinv1 = een / ux1
            xlinv1 = een / xl1
            uxinv2 = een / ux2
            xlinv2 = een / xl2
            plam = p0 + torch.matmul(P.T, lam)
            qlam = q0 + torch.matmul(Q.T, lam)
            gvec = torch.matmul(P, uxinv1) + torch.matmul(Q, xlinv1)
            GG = P * uxinv2.view(1, -1) - Q * xlinv2.view(1, -1)
            dpsidx = plam / ux2 - qlam / xl2
            delx = dpsidx - epsvecn / (x - alfa) + epsvecn / (beta - x)
            dely = c + d * y - lam - epsvecm / y
            delz = a0 - torch.matmul(a.T, lam) - epsi / z
            dellam = gvec - a * z - y - b + epsvecm / lam
            diagx = plam / ux3 + qlam / xl3
            diagx = 2 * diagx + xsi / (x - alfa) + eta / (beta - x)
            diagxinv = een / diagx
            diagy = d + mu / y
            diagyinv = eem / diagy
            diaglam = s / lam
            diaglamyi = diaglam + diagyinv

            # Solve system of equations
            if m < n:
                blam = dellam + dely / diagy - torch.matmul(GG, (delx / diagx))
                bb = torch.cat((blam, delz), dim=0)
                Alam = torch.diag(diaglamyi.view(-1)) + torch.matmul(GG * diagxinv.view(1, -1), GG.T)
                AAr1 = torch.cat((Alam, a), dim=1)
                AAr2 = torch.cat((a.T, -zet / z), dim=1)
                AA = torch.cat((AAr1, AAr2), dim=0)
                solut = torch.linalg.solve(AA, bb)
                dlam = solut[0:m]
                dz = solut[m:m + 1]
                dx = -delx / diagx - torch.matmul(GG.T, dlam) / diagx
            else:
                diaglamyiinv = eem / diaglamyi
                dellamyi = dellam + dely / diagy
                Axx = torch.diag(diagx.view(-1)) + torch.matmul(GG.T * diaglamyiinv.view(1, -1), GG)
                azz = zet / z + torch.matmul(a.T, (a / diaglamyi))
                axz = torch.matmul(-GG.T, (a / diaglamyi))
                bx = delx + torch.matmul(GG.T, (dellamyi / diaglamyi))
                bz = delz - torch.matmul(a.T, (dellamyi / diaglamyi))
                AAr1 = torch.cat((Axx, axz), dim=1)
                AAr2 = torch.cat((axz.T, azz), dim=1)
                AA = torch.cat((AAr1, AAr2), dim=0)
                bb = torch.cat((-bx, -bz), dim=0)
                solut = torch.linalg.solve(AA, bb)
                dx = solut[0:n]
                dz = solut[n:n + 1]
                dlam = torch.matmul(GG, dx) / diaglamyi - dz * (a / diaglamyi) + dellamyi / diaglamyi

            dy = -dely / diagy + dlam / diagy
            dxsi = -xsi + epsvecn / (x - alfa) - (xsi * dx) / (x - alfa)
            deta = -eta + epsvecn / (beta - x) + (eta * dx) / (beta - x)
            dmu = -mu + epsvecm / y - (mu * dy) / y
            dzet = -zet + epsi / z - zet * dz / z
            ds = -s + epsvecm / lam - (s * dlam) / lam
            xx = torch.cat((y, z, lam, xsi, eta, mu, zet, s), dim=0)
            dxx = torch.cat((dy, dz, dlam, dxsi, deta, dmu, dzet, ds), dim=0)

            # Step length determination
            stepxx = -1.01 * dxx / xx
            stmxx = torch.max(stepxx)
            stepalfa = -1.01 * dx / (x - alfa)
            stmalfa = torch.max(stepalfa)
            stepbeta = 1.01 * dx / (beta - x)
            stmbeta = torch.max(stepbeta)
            stmalbe = torch.maximum(stmalfa, stmbeta)
            stmalbexx = torch.maximum(stmalbe, stmxx)
            stminv = torch.maximum(stmalbexx, torch.tensor(1.0, dtype=dtype, device=device))
            steg = 1.0 / stminv

            # Update variables
            xold = x.clone()
            yold = y.clone()
            zold = z.clone()
            lamold = lam.clone()
            xsiold = xsi.clone()
            etaold = eta.clone()
            muold = mu.clone()
            zetold = zet.clone()
            sold = s.clone()
            
            itto = 0
            resinew = 2 * residunorm

            while (resinew > residunorm) and (itto < 50):
                itto += 1
                x = xold + steg * dx
                y = yold + steg * dy
                z = zold + steg * dz
                lam = lamold + steg * dlam
                xsi = xsiold + steg * dxsi
                eta = etaold + steg * deta
                mu = muold + steg * dmu
                zet = zetold + steg * dzet
                s = sold + steg * ds
                ux1 = upp - x
                xl1 = x - low
                ux2 = ux1 * ux1
                xl2 = xl1 * xl1
                uxinv1 = een / ux1
                xlinv1 = een / xl1
                plam = p0 + torch.matmul(P.T, lam)
                qlam = q0 + torch.matmul(Q.T, lam)
                gvec = torch.matmul(P, uxinv1) + torch.matmul(Q, xlinv1)
                dpsidx = plam / ux2 - qlam / xl2
                rex = dpsidx - xsi + eta
                rey = c + d * y - mu - lam
                rez = a0 - zet - torch.matmul(a.T, lam)
                relam = gvec - a * z - y + s - b
                rexsi = xsi * (x - alfa) - epsvecn
                reeta = eta * (beta - x) - epsvecn
                remu = mu * y - epsvecm
                rezet = zet * z - epsi
                res = lam * s - epsvecm
                residu1 = torch.cat((rex, rey, rez), dim=0)
                residu2 = torch.cat((relam, rexsi, reeta, remu, rezet, res), dim=0)
                residu = torch.cat((residu1, residu2), dim=0)
                resinew = torch.sqrt(torch.matmul(residu.T, residu)).item()
                steg = steg / 2
            residunorm = resinew
            residumax = torch.max(torch.abs(residu)).item()
            steg = 2 * steg

        epsi = 0.1 * epsi

    xmma = x.clone()
    ymma = y.clone()
    zmma = z.clone()
    lamma = lam
    xsimma = xsi
    etamma = eta
    mumma = mu
    zetmma = zet
    smma = s

    return xmma, ymma, zmma, lamma, xsimma, etamma, mumma, zetmma, smma

def kktcheck(m: int, n: int, x: torch.Tensor, y: torch.Tensor, z: float, lam: torch.Tensor, xsi: torch.Tensor, 
             eta: torch.Tensor, mu: torch.Tensor, zet: float, s: torch.Tensor, xmin: torch.Tensor, xmax: torch.Tensor, 
             df0dx: torch.Tensor, fval: torch.Tensor, dfdx: torch.Tensor, a0: float, a: torch.Tensor, c: torch.Tensor, 
             d: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, float, float]:
    
    """
    Evaluate the residuals for the Karush-Kuhn-Tucker (KKT) conditions of a nonlinear programming problem.
    """
    
    device = x.device
    dtype = x.dtype

    # Create vectors of ones for calculations
    een = torch.ones((n, 1), dtype=dtype, device=device)
    eem = torch.ones((m, 1), dtype=dtype, device=device)

    # Calculate residuals for the objective and constraints
    rex = df0dx + torch.matmul(dfdx.T, lam) - xsi + eta
    rey = c + d * y - mu - lam
    rez = a0 - zet - torch.matmul(a.T, lam)
    relam = fval - a * z - y + s
    rexsi = xsi * (x - xmin)
    reeta = eta * (xmax - x)
    remu = mu * y
    rezet = zet * z
    res = lam * s

    # Ensure rez and rezet are 2D tensors of shape (1, 1)
    if rez.dim() == 0:
        rez = rez.view(1, 1)
    elif rez.dim() == 1:
        rez = rez.view(1, 1)
    if rezet.dim() == 0:
        rezet = rezet.view(1, 1)
    elif rezet.dim() == 1:
        rezet = rezet.view(1, 1)

    # Concatenate residuals into a single vector
    residu1 = torch.cat((rex, rey, rez), dim=0)
    residu2 = torch.cat((relam, rexsi, reeta, remu, rezet, res), dim=0)
    residu = torch.cat((residu1, residu2), dim=0)

    # Calculate the L2 norm and maximum absolute value of the residuals
    residunorm = torch.sqrt(torch.matmul(residu.T, residu)).item()
    residumax = torch.max(torch.abs(residu)).item()

    return residu, residunorm, residumax

def raaupdate(xmma: torch.Tensor, xval: torch.Tensor, xmin: torch.Tensor, xmax: torch.Tensor, low: torch.Tensor, upp: torch.Tensor, 
              f0valnew: torch.Tensor, fvalnew: torch.Tensor, f0app: torch.Tensor, fapp: torch.Tensor, raa0: float, 
              raa: torch.Tensor, raa0eps: float, raaeps: torch.Tensor,  epsimin: float, **kwargs) -> Tuple[float, torch.Tensor]:
    
    """
    Update the parameters raa0 and raa during an inner iteration.
    """
    
    device = xmma.device
    dtype = xmma.dtype

    raacofmin = 1e-12
    eeem = torch.ones((raa.size(0), 1), dtype=dtype, device=device)
    eeen = torch.ones((xmma.size(0), 1), dtype=dtype, device=device)
    xmami = xmax - xmin
    xmamieps = 0.00001 * eeen
    xmami = torch.maximum(xmami, xmamieps)
    xxux = (xmma - xval) / (upp - xmma)
    xxxl = (xmma - xval) / (xmma - low)
    xxul = xxux * xxxl
    ulxx = (upp - low) / xmami
    raacof = torch.matmul(xxul.T, ulxx)
    raacof = torch.maximum(raacof, torch.tensor(raacofmin, dtype=dtype, device=device))
    f0appe = f0app + 0.5 * epsimin

    if torch.all(f0valnew > f0appe):
        deltaraa0 = (1.0 / raacof.item()) * (f0valnew.item() - f0app.item())
        zz0 = 1.1 * (raa0 + deltaraa0)
        zz0 = min(zz0, 10 * raa0)
        raa0 = zz0
    
    fappe = fapp + 0.5 * epsimin * eeem
    fdelta = fvalnew - fappe
    deltaraa = (1.0 / raacof.item()) * (fvalnew - fapp)
    zzz = 1.1 * (raa + deltaraa)
    zzz = torch.minimum(zzz, 10 * raa)
    mask = fdelta > 0
    raa[mask] = zzz[mask]
    
    return raa0, raa


def concheck(m: int, epsimin: float, f0app: torch.Tensor, f0valnew: torch.Tensor, fapp: torch.Tensor, fvalnew: torch.Tensor, **kwargs) -> int:
    
    """
    Check if the current approximations are conservative.
    """
    
    device = fapp.device
    dtype = fapp.dtype

    eeem = torch.ones((m, 1), dtype=dtype, device=device)
    f0appe = f0app + epsimin
    fappe = fapp + epsimin * eeem
    arr1 = torch.cat((f0appe.view(-1), fappe.view(-1)))
    arr2 = torch.cat((f0valnew.view(-1), fvalnew.view(-1)))
    
    if torch.all(arr1 >= arr2):
        conserv = 1
    else:
        conserv = 0
    
    return conserv


def asymp(outeriter: int, n: int, xval: torch.Tensor, xold1: torch.Tensor, xold2: torch.Tensor, xmin: torch.Tensor,
    xmax: torch.Tensor, low: torch.Tensor, upp: torch.Tensor, raa0: float, raa: torch.Tensor, raa0eps: float,
    raaeps: float, df0dx: torch.Tensor, dfdx: torch.Tensor, asyinit: float = 0.5, asydecr: float = 0.7, 
    asyincr: float = 1.2, asymin: float = 0.01, asymax: float = 10, **kwargs)-> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    
    """
    Calculate the parameters raa0, raa, low, and upp at the beginning of each outer iteration.
    """
    
    device = xval.device
    dtype = xval.dtype

    eeen = torch.ones((n, 1), dtype=dtype, device=device)
    xmami = xmax - xmin
    xmamieps = 0.00001 * eeen
    xmami = torch.maximum(xmami, xmamieps)
    raa0_val = torch.matmul(torch.abs(df0dx).T, xmami).item()
    raa0 = max(raa0eps, (0.1 / n) * raa0_val)
    raa = torch.matmul(torch.abs(dfdx), xmami)
    raa = torch.maximum(torch.tensor(raaeps, dtype=dtype, device=device), (0.1 / n) * raa)
    
    if outeriter <= 2:
        low = xval - asyinit * xmami
        upp = xval + asyinit * xmami
    else:
        xxx = (xval - xold1) * (xold1 - xold2)
        factor = eeen.clone()
        factor[xxx > 0] = asyincr
        factor[xxx < 0] = asydecr
        low = xval - factor * (xold1 - low)
        upp = xval + factor * (upp - xold1)
        lowmin = xval - asymax * xmami
        lowmax = xval - asymin * xmami
        uppmin = xval + asymin * xmami
        uppmax = xval + asymax * xmami
        low = torch.maximum(low, lowmin)
        low = torch.minimum(low, lowmax)
        upp = torch.minimum(upp, uppmax)
        upp = torch.maximum(upp, uppmin)
    
    return low, upp, raa0, raa