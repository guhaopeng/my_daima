import torch
from Stress_3D_Sensitivity_Comp import Stress_3D_Sensitivity_Comp

def test_original_l_shape():
    print("Testing original L-shape fallback...")
    nelx, nely, nelz = 4, 4, 4
    x = torch.ones(nelx * nely * nelz, dtype=torch.float64, requires_grad=True)
    
    # Not passing F_dense and fixeddof_in should trigger L-shape logic
    pnorm, mises = Stress_3D_Sensitivity_Comp(x, nelx, nely, nelz, pl=3.0, q=0.5, p=8.0)
    print(f"P-norm (Original L-shape): {pnorm.item():.4f}")
    
    pnorm.backward()
    print(f"Gradient sum (Original L-shape): {x.grad.sum().item():.4f}")
    print("Original L-shape test passed!\n")

def test_external_force_and_constraints():
    print("Testing external F_dense and fixeddof...")
    nelx, nely, nelz = 4, 4, 4
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    
    x = torch.ones(nelx * nely * nelz, dtype=torch.float64, requires_grad=True)
    
    # 1. Manually construct F_dense
    F_dense = torch.zeros(ndof, dtype=torch.float64)
    # Apply a force downwards in Y direction at some random node
    test_node_id = (nelx + 1) * (nely + 1) * (nelz + 1) // 2
    F_dense[3 * test_node_id + 1] = -5.0
    
    # 2. Manually construct fixeddof
    # Fix the first 4 nodes
    fixed_nodes = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    fixeddof_in = torch.cat([3 * fixed_nodes, 3 * fixed_nodes + 1, 3 * fixed_nodes + 2])
    
    pnorm, mises = Stress_3D_Sensitivity_Comp(
        x, nelx, nely, nelz, pl=3.0, q=0.5, p=8.0, 
        F_dense=F_dense, fixeddof_in=fixeddof_in
    )
    print(f"P-norm (External params): {pnorm.item():.4f}")
    
    pnorm.backward()
    print(f"Gradient sum (External params): {x.grad.sum().item():.4f}")
    print("External params test passed!")

if __name__ == "__main__":
    test_original_l_shape()
    test_external_force_and_constraints()