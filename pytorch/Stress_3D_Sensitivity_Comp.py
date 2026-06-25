import torch

def cg_solve(A, b, tol=1e-8, max_iter=5000):
    orig_dtype = b.dtype
    # 为保证数值稳定性（尤其是受力非常大时），将数据转换到 float64 求解
    if orig_dtype != torch.float64:
        A = A.to(torch.float64)
        b = b.to(torch.float64)

    b_norm = torch.norm(b)
    if b_norm == 0:
        return torch.zeros_like(b, dtype=orig_dtype)
    actual_tol = tol * b_norm

    # Check for NaN in input
    if torch.isnan(b).any() or torch.isinf(b).any():
        print("Warning: b contains NaN or Inf. Returning zeros.")
        return torch.zeros_like(b, dtype=orig_dtype)

    # 转换为 CSR 格式
    # 1. 速度显著提升 (CSR 的矩阵向量乘法极快)
    # 2. 保证了严格的数值正交性 (COO 格式的 SpMV 在 GPU 上使用原子加法，非确定性的舍入误差会导致 CG 发散，而 CSR 是确定性的)
    A_csr = A.to_sparse_csr()

    indices = A.indices()
    values = A.values()
    diag_mask = indices[0] == indices[1]
    diag_vals = values[diag_mask]
    n = b.size(0)
    diag = torch.ones(n, dtype=b.dtype, device=b.device)
    diag_indices = indices[0][diag_mask]
    diag.scatter_(0, diag_indices, diag_vals)
    
    diag[torch.abs(diag) < 1e-14] = 1e-14
    
    # Jacobi 预条件器
    M_inv = 1.0 / diag
    M_inv[torch.isinf(M_inv) | torch.isnan(M_inv)] = 1.0
    
    x = torch.zeros_like(b)
    # 使用 CSR 的 matmul，b 需要是 2D tensor，然后再 squeeze 回 1D
    r = b - torch.matmul(A_csr, x.unsqueeze(1)).squeeze(1)
    z = r * M_inv
    p = z.clone()
    rsold = torch.dot(r, z)
    
    for i in range(max_iter):
        Ap = torch.matmul(A_csr, p.unsqueeze(1)).squeeze(1)
        p_Ap = torch.dot(p, Ap)
        if p_Ap == 0:
            break
        alpha = rsold / p_Ap
        x = x + alpha * p
        r = r - alpha * Ap
        if torch.norm(r) < actual_tol:
            # print(f"PyTorch CSR CG 收敛于第 {i+1} 次迭代")
            return x.to(orig_dtype)
        z = r * M_inv
        rsnew = torch.dot(r, z)
        p = z + (rsnew / rsold) * p
        rsold = rsnew
        
    print(f"CG未收敛: 达到最大迭代次数 {max_iter}, 残差范数 {torch.norm(r).item()}")
    return x.to(orig_dtype)

class CGSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, K_vals, K_indices, K_shape, F, tolit, maxit):
        K = torch.sparse_coo_tensor(K_indices, K_vals, K_shape).coalesce()
        U = cg_solve(K, F, tol=tolit, max_iter=maxit)
        ctx.save_for_backward(K_vals, K_indices, U)
        ctx.K_shape = K_shape
        ctx.tolit = tolit
        ctx.maxit = maxit
        return U

    @staticmethod
    def backward(ctx, grad_U):
        K_vals, K_indices, U = ctx.saved_tensors
        K = torch.sparse_coo_tensor(K_indices, K_vals, ctx.K_shape).coalesce()
        lambda_ = cg_solve(K, grad_U, tol=ctx.tolit, max_iter=ctx.maxit)
        row, col = K_indices
        K_vals_grad = - lambda_[row] * U[col]
        return K_vals_grad, None, None, None, None, None

def Stress_3D_Sensitivity_Comp(x, nelx, nely, nelz, pl, q, p, F_dense=None, fixeddof_in=None):
    '''计算 3D 应力并支持 PyTorch Autograd'''
    device = x.device
    dtype = x.dtype
    
    KE, B, D = brick_stiffnessMatrix(device, dtype)
    E0 = 1.0                                                    
    Emin = 1e-3                                                 

    x = x.view(-1)
    nele = nelx * nely * nelz                                   
    if x.numel() != nele:
        raise ValueError("x 的长度必须等于 nelx*nely*nelz")

    # ---------------- 原始载荷与约束 (已注释) ----------------
    # kl = torch.arange(0, nelz + 1, dtype=torch.int64, device=device)
    # loadnid = kl * (nelx + 1) * (nely + 1) + nelx * (nely + 1) + (nely + 1)
    # loaddof = 3 * loadnid - 2

    # jf = torch.arange(0, nely + 1, dtype=torch.int64, device=device)
    # kf = torch.arange(0, nelz + 1, dtype=torch.int64, device=device)
    # JF, KF = torch.meshgrid(jf, kf, indexing='xy')
    # fixednid = KF * (nelx + 1) * (nely + 1) + (nely + 1 - JF)
    # fixednid = fixednid.t().reshape(-1)
    # fixeddof = torch.cat((3 * fixednid - 1, 3 * fixednid - 2, 3 * fixednid - 3))

    # ---------------- 3D-L 梁的载荷与约束或外部传入 ----------------
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)

    if F_dense is not None and fixeddof_in is not None:
        F = F_dense.clone()
        fixeddof = fixeddof_in.to(device=device, dtype=torch.int64)
    else:
        # 节点索引 (0-based): node_id(x, y, z) = z * (nelx+1)*(nely+1) + x * (nely+1) + y
        # X 从 0 到 nelx (从左到右), Y 从 0 到 nely (从上到下), Z 从 0 到 nelz
        
        # 1. 约束 (Fixed DOFs): 顶部面的左半部分 (Y=0, X=0 to nelx//2, all Z)
        # 这完全符合 L-shape 示意图中左侧顶部红色 "Fixed" 区域
        fixed_nodes = []
        for z in range(nelz + 1):
            for x_idx in range((nelx // 2) + 1):
                node_id = z * (nelx + 1) * (nely + 1) + x_idx * (nely + 1) + 0
                fixed_nodes.append(node_id)
        fixed_nodes_tensor = torch.tensor(fixed_nodes, dtype=torch.int64, device=device)
        fixeddof = torch.cat([3 * fixed_nodes_tensor, 3 * fixed_nodes_tensor + 1, 3 * fixed_nodes_tensor + 2])
        
        # 2. 载荷 (Load DOFs): 右端面的顶部边缘向下受力 (X=nelx, Y=nely//2, all Z)
        # 这完全符合 L-shape 示意图中右侧中间红色 "Force" 箭头
        # 为避免点载荷应力集中，像原代码一样分布在 Y=nely//2 及相邻共 3 个节点
        load_nodes = []
        for z in range(nelz + 1):
            for y_offset in range(3): # Y = nely//2, nely//2 + 1, nely//2 + 2
                y_idx = (nely // 2) + y_offset
                if y_idx <= nely:
                    node_id = z * (nelx + 1) * (nely + 1) + nelx * (nely + 1) + y_idx
                    load_nodes.append(node_id)
        load_nodes_tensor = torch.tensor(load_nodes, dtype=torch.int64, device=device)
        # 向下受力，即 Y 自由度 (3*node_id + 1)
        loaddof = 3 * load_nodes_tensor + 1

        F = torch.zeros(ndof, dtype=dtype, device=device)                                  
        F[loaddof] = -1.0
    # -------------------------------------------------------------------------


    freedofs_set = set(range(ndof)) - set(fixeddof.tolist())
    freedofs_list = sorted(list(freedofs_set))
    freedofs = torch.tensor(freedofs_list, dtype=torch.int64, device=device)

    nodegrd = torch.arange(1, (nely + 1) * (nelx + 1) + 1, dtype=torch.int64, device=device).view(nelx + 1, nely + 1).T   
    nodeids = nodegrd[:-1, :-1].T.reshape(-1, 1)  
    nodeidz = torch.arange(0, nelz * (nely + 1) * (nelx + 1), (nely + 1) * (nelx + 1), dtype=torch.int64, device=device)  
    nodeids = nodeids.repeat(1, nodeidz.size(0)) + nodeidz.repeat(nodeids.size(0), 1)  
    nodeids = nodeids.T.reshape(-1, 1)             
    edof_vec = 3 * nodeids + 1                              
    local_offsets = torch.tensor([0, 1, 2, 3 * nely + 3, 3 * nely + 4, 3 * nely + 5, 3 * nely, 3 * nely + 1, 3 * nely + 2, -3, -2, -1], dtype=torch.int64, device=device)  
    local_offsets = torch.cat((local_offsets, 3 * (nely + 1) * (nelx + 1) + local_offsets))  
    edof_mat = edof_vec.repeat(1, 24) + local_offsets.repeat(nele, 1)  
    edof_mat = edof_mat - 1                      

    iK = edof_mat.repeat_interleave(24, dim=1).view(-1)
    jK = edof_mat.repeat(1, 24).view(-1)
    
    Ee = Emin + (x ** pl) * (E0 - Emin)                          
    sK = (Ee.view(-1, 1) * KE.T.reshape(1, -1)).view(-1)
    
    is_free = torch.ones(ndof, dtype=torch.bool, device=device)
    is_free[fixeddof] = False
    free_mask = is_free[iK] & is_free[jK]
    
    iK_free = iK[free_mask]
    jK_free = jK[free_mask]
    sK_free = sK[free_mask]
    
    iK_fixed = fixeddof
    jK_fixed = fixeddof
    sK_fixed = torch.ones_like(fixeddof, dtype=dtype)
    
    K_indices = torch.stack([torch.cat([iK_free, iK_fixed]), torch.cat([jK_free, jK_fixed])], dim=0)
    K_vals = torch.cat([sK_free, sK_fixed])

    tolit = 1e-8 
    maxit = 5000
    
    U = CGSolve.apply(K_vals, K_indices, (ndof, ndof), F, tolit, maxit)

    # Compute Stresses
    ue = U[edof_mat] # Shape: (nele, 24)
    # temp = (x ** q) * (D @ B @ ue)
    # D is (6, 6), B is (6, 24), DB is (6, 24)
    DB = torch.matmul(D, B) # (6, 24)
    # ue is (nele, 24), we want (nele, 6)
    stress_base = torch.matmul(ue, DB.T) # (nele, 6)
    S = (x ** q).view(-1, 1) * stress_base

    S11 = S[:, 0]
    S22 = S[:, 1]
    S33 = S[:, 2]
    S12 = S[:, 3]
    S23 = S[:, 4]
    S13 = S[:, 5]

    MISES = torch.sqrt(0.5 * ((S11 - S22) ** 2 + (S11 - S33) ** 2 + (S22 - S33) ** 2 + 6.0 * (S12 ** 2 + S23 ** 2 + S13 ** 2)))
    
    pnorm_base = torch.sum(MISES ** p)  
    pnorm = pnorm_base ** (1.0 / p)

    return pnorm, MISES

def brick_stiffnessMatrix(device, dtype):
    nu = 0.3 
    D = 1/((1+nu) * (1-2*nu)) * torch.tensor([[1-nu,nu,nu,0,0,0],  
                                       [nu,1-nu,nu,0,0,0],
                                       [nu,nu,1-nu,0,0,0],
                                       [0,0,0,0.5-nu,0,0],
                                       [0,0,0,0,0.5-nu,0],
                                       [0,0,0,0,0,0.5-nu]], dtype=dtype, device=device)
    A = torch.tensor([[32, 6, -8, 6, -6, 4, 3, -6, -10, 3, -3, -3, -4, -8],  
                  [-48, 0, 0, -24, 24, 0, 0, 0, 12, -12, 0, 12, 12, 12]], dtype=dtype, device=device)
    
    k = (1.0 / 144.0) * (A.T @ torch.tensor([[1.0], [nu]], dtype=dtype, device=device))  
    k = k.view(-1)  
    K1 = torch.tensor([[k[0], k[1], k[1], k[2], k[4], k[4]], 
                   [k[1], k[0], k[1], k[3], k[5], k[6]], 
                   [k[1], k[1], k[0], k[3], k[6], k[5]], 
                   [k[2], k[3], k[3], k[0], k[7], k[7]], 
                   [k[4], k[5], k[6], k[7], k[0], k[1]], 
                   [k[4], k[6], k[5], k[7], k[1], k[0]]], dtype=dtype, device=device)  

    K2 = torch.tensor([[k[8], k[7], k[11], k[5], k[3], k[6]], 
                   [k[7], k[8], k[11], k[4], k[2], k[4]], 
                   [k[9], k[9], k[12], k[6], k[3], k[5]], 
                   [k[5], k[4], k[10], k[8], k[1], k[9]], 
                   [k[3], k[2], k[4], k[1], k[8], k[11]], 
                   [k[10], k[3], k[5], k[11], k[9], k[12]]], dtype=dtype, device=device)  

    K3 = torch.tensor([[k[5], k[6], k[3], k[8], k[11], k[7]], 
                   [k[6], k[5], k[3], k[9], k[12], k[9]], 
                   [k[4], k[4], k[2], k[7], k[11], k[8]], 
                   [k[8], k[9], k[1], k[5], k[10], k[4]], 
                   [k[11], k[12], k[9], k[10], k[5], k[3]],
                   [k[1], k[11], k[8], k[3], k[4], k[2]]], dtype=dtype, device=device)  

    K4 = torch.tensor([[k[13], k[10], k[10], k[12], k[9], k[9]], 
                   [k[10], k[13], k[10], k[11], k[8], k[7]], 
                   [k[10], k[10], k[13], k[11], k[7], k[8]], 
                   [k[12], k[11], k[11], k[13], k[6], k[6]], 
                   [k[9], k[8], k[7], k[6], k[13], k[10]], 
                   [k[9], k[7], k[8], k[6], k[10], k[13]]], dtype=dtype, device=device)  

    K5 = torch.tensor([[k[0], k[1], k[7], k[2], k[4], k[3]], 
                   [k[1], k[0], k[7], k[3], k[5], k[10]], 
                   [k[7], k[7], k[0], k[4], k[10], k[5]], 
                   [k[2], k[3], k[4], k[0], k[7], k[1]], 
                   [k[4], k[5], k[10], k[7], k[0], k[7]], 
                   [k[3], k[10], k[5], k[1], k[7], k[0]]], dtype=dtype, device=device)  

    K6 = torch.tensor([[k[13], k[10], k[6], k[12], k[9], k[11]], 
                   [k[10], k[13], k[6], k[11], k[8], k[1]], 
                   [k[6], k[6], k[13], k[9], k[1], k[8]], 
                   [k[12], k[11], k[9], k[13], k[6], k[10]], 
                   [k[9], k[8], k[1], k[6], k[13], k[6]], 
                   [k[11], k[1], k[8], k[10], k[6], k[13]]], dtype=dtype, device=device)  

    KE = (1.0 / ((nu + 1.0) * (1.0 - 2.0 * nu))) * torch.cat([
        torch.cat([K1, K2, K3, K4], dim=1),
        torch.cat([K2.T, K5, K6, K3.T], dim=1),
        torch.cat([K3.T, K6, K5.T, K2.T], dim=1),
        torch.cat([K4, K3, K2, K1.T], dim=1)
    ], dim=0)
 
    B_1 = torch.tensor([[-0.044658, 0.0, 0.0, 0.044658, 0.0, 0.0, 0.16667, 0.0], 
                    [0.0, -0.044658, 0.0, 0.0, -0.16667, 0.0, 0.0, 0.16667], 
                    [0.0, 0.0, -0.044658, 0.0, 0.0, -0.16667, 0.0, 0.0], 
                    [-0.044658, -0.044658, 0.0, -0.16667, 0.044658, 0.0, 0.16667, 0.16667], 
                    [0.0, -0.044658, -0.044658, 0.0, -0.16667, -0.16667, 0.0, -0.62201], 
                    [-0.044658, 0.0, -0.044658, -0.16667, 0.0, 0.044658, -0.62201, 0.0]], dtype=dtype, device=device)  
    B_2 = torch.tensor([[0.0, -0.16667, 0.0, 0.0, -0.16667, 0.0, 0.0, 0.16667], 
                    [0.0, 0.0, 0.044658, 0.0, 0.0, -0.16667, 0.0, 0.0], 
                    [-0.62201, 0.0, 0.0, -0.16667, 0.0, 0.0, 0.044658, 0.0], 
                    [0.0, 0.044658, -0.16667, 0.0, -0.16667, -0.16667, 0.0, -0.62201], 
                    [0.16667, 0.0, -0.16667, 0.044658, 0.0, 0.044658, -0.16667, 0.0], 
                    [0.16667, -0.16667, 0.0, -0.16667, 0.044658, 0.0, -0.16667, 0.16667]], dtype=dtype, device=device)  
    B_3 = torch.tensor([[0.0, 0.0, 0.62201, 0.0, 0.0, -0.62201, 0.0, 0.0], 
                    [-0.62201, 0.0, 0.0, 0.62201, 0.0, 0.0, 0.16667, 0.0], 
                    [0.0, 0.16667, 0.0, 0.0, 0.62201, 0.0, 0.0, 0.16667], 
                    [0.16667, 0.0, 0.62201, 0.62201, 0.0, 0.16667, -0.62201, 0.0], 
                    [0.16667, -0.62201, 0.0, 0.62201, 0.62201, 0.0, 0.16667, 0.16667], 
                    [0.0, 0.16667, 0.62201, 0.0, 0.62201, 0.16667, 0.0, -0.62201]], dtype=dtype, device=device)       
    B = torch.cat((B_1, B_2, B_3), dim=1)  
    return KE, B, D
                
