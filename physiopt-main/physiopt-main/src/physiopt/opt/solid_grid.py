from typing import Tuple

import torch
from torch import Tensor

from torchfem.base import FEM
from torchfem.elements import Hexa1, Hexa2, Tetra1, Tetra2
from torchfem.materials import IsotropicElasticity3D
from torchfem.sparse import sparse_solve

G_CONSTANT = 9.81


class SolidGrid(FEM):
    """
    Defines and solves for static equilibrium given a sparse FEM volume.
    Adapted from: https://github.com/meyer-nils/torch-fem/blob/main/src/torchfem/base.py
    """

    def __init__(
        self, nodes: Tensor, elements: Tensor, material: IsotropicElasticity3D
    ):
        """Initialize the solid FEM problem."""

        super().__init__(nodes, elements, material)

        # Only support hexahedral FEM + 3D + custom IsotropicElasticity3D
        assert len(elements[0]) == 8
        assert self.n_dim == 3
        assert isinstance(material, IsotropicElasticity3D)

        # Set element type depending on number of nodes per element
        if len(elements[0]) == 4:
            self.etype = Tetra1()
        elif len(elements[0]) == 8:
            self.etype = Hexa1()
        elif len(elements[0]) == 10:
            self.etype = Tetra2()
        elif len(elements[0]) == 20:
            self.etype = Hexa2()
        else:
            raise ValueError("Element type not supported.")

        # Set element type specific sizes
        self.n_strains = 6
        self.n_int = len(self.etype.iweights())

        # Initialize external strain
        self.ext_strain = torch.zeros(self.n_elem, self.n_strains, device=nodes.device)

    def D(self, B: Tensor, nodes: Tensor) -> Tensor:
        """Element gradient operator"""
        zeros = torch.zeros(self.n_elem, self.etype.nodes, device=nodes.device)
        shape = [self.n_elem, -1]
        D0 = torch.stack([B[:, 0, :], zeros, zeros], dim=-1).reshape(shape)
        D1 = torch.stack([zeros, B[:, 1, :], zeros], dim=-1).reshape(shape)
        D2 = torch.stack([zeros, zeros, B[:, 2, :]], dim=-1).reshape(shape)
        D3 = torch.stack([zeros, B[:, 2, :], B[:, 1, :]], dim=-1).reshape(shape)
        D4 = torch.stack([B[:, 2, :], zeros, B[:, 0, :]], dim=-1).reshape(shape)
        D5 = torch.stack([B[:, 1, :], B[:, 0, :], zeros], dim=-1).reshape(shape)
        return torch.stack([D0, D1, D2, D3, D4, D5], dim=1)

    def compute_k(self, detJ: Tensor, DCD: Tensor) -> Tensor:
        """Element stiffness matrix"""
        return torch.einsum("j,jkl->jkl", detJ, DCD)

    def compute_f(self, detJ: Tensor, D: Tensor, S: Tensor) -> Tensor:
        """Element internal force vector."""
        return torch.einsum("j,jkl,jk->jl", detJ, D, S)

    def k0(self) -> Tensor:
        """Compute element stiffness matrix for zero strain."""
        e = torch.zeros(2, self.n_int, self.n_elem, self.n_strains, device=self.device)
        s = torch.zeros(2, self.n_int, self.n_elem, self.n_strains, device=self.device)
        a = torch.zeros(
            2, self.n_int, self.n_elem, self.material.n_state, device=self.device
        )
        du = torch.zeros_like(self.nodes)
        dde0 = torch.zeros(self.n_elem, self.n_strains, device=self.device)
        self.K = torch.empty(0, device=self.device)
        k, _, _ = self.integrate(e, s, a, 1, du, dde0)
        return k

    # Modified version to take into
    def integrate(
        self,
        eps: Tensor,
        sig: Tensor,
        sta: Tensor,
        n: int,
        du: Tensor,
        de0: Tensor,
        use_gravity: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Perform numerical integrations for element stiffness matrix."""
        # Reshape variables
        nodes = self.nodes[self.elements, :]
        du = du.reshape((-1, self.n_dim))[self.elements, :].reshape(self.n_elem, -1)

        # Initialize nodal force and stiffness
        N_nod = self.etype.nodes
        f = torch.zeros(self.n_elem, self.n_dim * N_nod, device=self.device)
        f_gravity = torch.zeros(self.n_elem, self.n_dim * N_nod, device=self.device)
        if self.K.numel() == 0 or not self.material.n_state == 0:
            k = torch.zeros(
                (self.n_elem, self.n_dim * N_nod, self.n_dim * N_nod),
                device=self.device,
            )
        else:
            k = torch.empty(0, device=self.device)

        for i, (w, xi) in enumerate(zip(self.etype.iweights(), self.etype.ipoints())):
            # Compute gradient operators
            b = self.etype.B(xi).to(self.device)
            if b.shape[0] == 1:
                dx = nodes[:, 1] - nodes[:, 0]
                J = 0.5 * torch.linalg.norm(dx, dim=1)[:, None, None]
            else:
                J = torch.einsum("jk,mkl->mjl", b, nodes)
            detJ = torch.linalg.det(J)
            if torch.any(detJ <= 0.0):
                raise Exception("Negative Jacobian. Check element numbering.")
            B = torch.einsum("jkl,lm->jkm", torch.linalg.inv(J), b)
            D = self.D(B, nodes)

            # Evaluate material response
            de = torch.einsum("jkl,jl->jk", D, du) - de0
            eps[n, i], sig[n, i], sta[n, i], ddsdde = self.material.step(
                de, eps[n - 1, i], sig[n - 1, i], sta[n - 1, i]
            )

            # Compute element internal forces
            # i.e., propagate from element to nodes (with weights)
            # f has shape [n_elements, n_nodes_per_element * 3]
            # NOTE: it will be aggregated further down with assemble_force
            f += w * self.compute_f(detJ, D, sig[n, i].clone())

            # Compute element stiffness matrix
            if self.K.numel() == 0 or not self.material.n_state == 0:
                DCD = torch.einsum("jkl,jlm,jkn->jmn", ddsdde, D, D)
                k += w * self.compute_k(detJ, DCD)

            # Compute element gravity force
            y_down_vec = (
                torch.tensor([0.0, -1.0, 0.0], device="cuda")[None, None, :]
                .repeat((1, N_nod, 1))
                .reshape((1, self.n_dim * N_nod))
            )
            if use_gravity:
                f_gravity += (
                    w
                    * self.material.rho[:, None]
                    * G_CONSTANT
                    * y_down_vec
                    * detJ[:, None]
                )

        return k, f, f_gravity

    def solve(
        self,
        increments: Tensor = torch.tensor([0.0, 1.0]),
        max_iter: int = 10,
        rtol: float = 1e-8,
        atol: float = 1e-6,
        stol: float = 1e-8,
        verbose: bool = False,
        return_intermediate: bool = False,
        aggregate_integration_points: bool = True,
        use_gravity: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Solve the FEM problem with the Newton-Raphson method."""
        # Number of increments
        N = len(increments)

        # Indexes of constrained and unconstrained degrees of freedom
        con = torch.nonzero(self.constraints.ravel(), as_tuple=False).ravel()

        # Initialize variables to be computed
        epsilon = torch.zeros(
            N, self.n_int, self.n_elem, self.n_strains, device=self.device
        )
        sigma = torch.zeros(
            N, self.n_int, self.n_elem, self.n_strains, device=self.device
        )
        state = torch.zeros(
            N, self.n_int, self.n_elem, self.material.n_state, device=self.device
        )
        all_f_int = torch.zeros(N, self.n_nod, self.n_dim, device=self.device)
        all_f_ext = torch.zeros(N, self.n_nod, self.n_dim, device=self.device)
        u = torch.zeros(N, self.n_nod, self.n_dim, device=self.device)

        # Initialize global stiffness matrix
        self.K = torch.empty(0)

        # Initialize displacement increment
        du = torch.zeros_like(self.nodes).ravel()

        # Incremental loading
        for n in range(1, N):
            assert N == 2
            # Increment size
            inc = increments[n] - increments[n - 1]

            # Load increment
            F_ext = increments[n] * self.forces.ravel()
            DU = inc * self.displacements.clone().ravel()
            DE = inc * self.ext_strain

            # Newton-Raphson iterations
            for i in range(max_iter + 1):
                assert max_iter == 1
                du[con] = DU[con]

                # Element-wise integration
                k, f_int, f_gravity = self.integrate(
                    epsilon, sigma, state, n, du, DE, use_gravity=use_gravity
                )

                # Assemble global stiffness matrix and internal force vector (if needed)
                if self.K.numel() == 0 or not self.material.n_state == 0:
                    self.K = self.assemble_stiffness(k, con)
                F_int = self.assemble_force(f_int)
                F_gravity = self.assemble_force(f_gravity)

                # Compute residual
                residual = F_int - F_ext - F_gravity
                residual[con] = 0.0
                res_norm = torch.linalg.norm(residual)

                # Save initial residual
                if i == 0:
                    res_norm0 = res_norm

                # Print iteration information
                if verbose:
                    print(f"Increment {n} | Iteration {i+1} | Residual: {res_norm:.5e}")

                # Check convergence
                if res_norm < rtol * res_norm0 or res_norm < atol or i == max_iter:
                    break

                # Solve for displacement increment
                du -= sparse_solve(self.K, residual, stol)

            # if res_norm > rtol * res_norm0 and res_norm > atol:
            #     raise Exception("Newton-Raphson iteration did not converge.")

            # Update increment
            all_f_int[n] = F_int.reshape((-1, self.n_dim))
            all_f_ext[n] = (F_ext + F_gravity).reshape((-1, self.n_dim))
            u[n] = u[n - 1] + du.reshape((-1, self.n_dim))

        # Aggregate integration points as mean
        if aggregate_integration_points:
            epsilon = epsilon.mean(dim=1)
            sigma = sigma.mean(dim=1)
            state = state.mean(dim=1)

        if return_intermediate:
            # Return all intermediate values
            return u, all_f_int, all_f_ext, sigma, epsilon, state
        else:
            # Return only the final values
            return (
                u[-1],
                all_f_int[-1],
                all_f_ext[-1],
                sigma[-1],
                epsilon[-1],
                state[-1],
            )
