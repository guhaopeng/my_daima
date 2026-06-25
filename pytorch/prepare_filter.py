import torch

def prepare_filter(rmin, nelx, nely, nelz, device='cpu', dtype=torch.float32):
    nele = nelx * nely * nelz
    iH = []
    jH = []
    sH = []
    R = int(torch.ceil(torch.tensor(rmin)).item()) - 1
    for k1 in range(1, nelz + 1):
        for i1 in range(1, nelx + 1):
            for j1 in range(1, nely + 1):
                e1 = (k1 - 1) * nelx * nely + (i1 - 1) * nely + j1
                k2_min = max(k1 - R, 1)
                k2_max = min(k1 + R, nelz)
                i2_min = max(i1 - R, 1)
                i2_max = min(i1 + R, nelx)
                j2_min = max(j1 - R, 1)
                j2_max = min(j1 + R, nely)
                for k2 in range(k2_min, k2_max + 1):
                    for i2 in range(i2_min, i2_max + 1):
                        for j2 in range(j2_min, j2_max + 1):
                            e2 = (k2 - 1) * nelx * nely + (i2 - 1) * nely + j2
                            w = rmin - ((i1 - i2) ** 2 + (j1 - j2) ** 2 + (k1 - k2) ** 2) ** 0.5
                            if w > 0:
                                iH.append(e1 - 1)
                                jH.append(e2 - 1)
                                sH.append(w)
    
    iH_tensor = torch.tensor(iH, dtype=torch.long, device=device)
    jH_tensor = torch.tensor(jH, dtype=torch.long, device=device)
    sH_tensor = torch.tensor(sH, dtype=dtype, device=device)
    
    indices = torch.stack((iH_tensor, jH_tensor))
    H = torch.sparse_coo_tensor(indices, sH_tensor, (nele, nele), device=device).coalesce()
    
    # Compute sum along rows for Hs
    Hs = torch.sparse.sum(H, dim=1).to_dense()
    
    return Hs, H
