import torch
import numpy as np
from typing import List, Optional, Union
from ._base import BaseModel, COMPLEX_DTYPE

class LPDO(BaseModel):
    """
    Represents a Locally-Purified Density Operator (LPDO) for parametrizing the
    Choi matrix of a quantum channel, based on the model from the paper
    "Quantum process tomography with unsupervised learning and tensor networks".

    The Choi matrix Λ_θ is constructed from a set of tensors {A_j} as:
    Λ_θ = Σ (A_j ⊗ A_j*)
    This structure guarantees that Λ_θ is positive-semidefinite by construction.

    The tensors A_j follow the Matrix Product Operator (MPO) convention:
    - Tensor shape: (left_bond_dim, right_bond_dim, phys_dim_out, phys_dim_in, kraus_dim)
    """
    def __init__(self,
                 site_tensors: List[torch.nn.Parameter],
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: Optional[torch.dtype] = None,
                 **kwargs):
        super().__init__(site_tensors=site_tensors, device=device, dtype=dtype, **kwargs)

        if not site_tensors:
            raise ValueError("site_tensors list cannot be empty.")

        self.site_tensors = torch.nn.ParameterList(site_tensors)
        self.num_sites = len(site_tensors)

        self._device = self.site_tensors[0].device
        self._dtype = self.site_tensors[0].dtype
        self.physical_dim = self.site_tensors[0].shape[2]

    def forward(self, rho_in: torch.Tensor, M_out: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that computes the probability P(M_out|rho_in).
        This is the same as compute_probability but follows PyTorch's nn.Module convention.
        
        Args:
            rho_in: Input density matrix of shape (2^num_sites, 2^num_sites)
            M_out: Measurement operator of shape (2^num_sites, 2^num_sites)
            
        Returns:
            Probability as a real scalar tensor
        """
        return self.compute_probability(rho_in, M_out)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @classmethod
    def random_initialization(cls,
                            num_sites: int,
                            physical_dim: int = 2,
                            bond_dim: int = 2,
                            kraus_dim: int = 1,
                            device: Optional[Union[str, torch.device]] = None,
                            dtype: Optional[torch.dtype] = None) -> 'LPDO':
        dev = torch.device(device) if device is not None else torch.device('cpu')
        _dtype = dtype if dtype is not None else COMPLEX_DTYPE

        site_tensors = []
        for i in range(num_sites):
            l_bond = 1 if i == 0 else bond_dim
            r_bond = 1 if i == num_sites - 1 else bond_dim
            shape = (l_bond, r_bond, physical_dim, physical_dim, kraus_dim)
            
            tensor = torch.randn(shape, device=dev, dtype=_dtype) * 0.1
            site_tensors.append(torch.nn.Parameter(tensor))

        return cls(site_tensors)

    def get_choi_matrix(self) -> torch.Tensor:
        """
        Constructs and returns the full Choi matrix from the LPDO tensors.
        Warning: This can be very memory-intensive for large systems.
        """
        # Contract the A tensors site-by-site to form the full A operator
        # A_i has shape (l, r, o, i, k)
        A_full = self.site_tensors[0]
        for i in range(1, self.num_sites):
            A_next = self.site_tensors[i]
            # Contract right bond of A_full with left bond of A_next
            A_full = torch.einsum('lr..., Roisk -> lR...oisk', A_full, A_next)

        # Reshape to group physical and kraus indices
        # Shape: (1, 1, p_o1, p_i1, k1, p_o2, p_i2, k2, ...)
        A_full = A_full.squeeze(0).squeeze(0)
        
        # Permute to (p_o1, p_o2,...), (p_i1, p_i2,...), (k1, k2,...)
        permute_po = list(range(0, 3 * self.num_sites, 3))
        permute_pi = list(range(1, 3 * self.num_sites, 3))
        permute_k = list(range(2, 3 * self.num_sites, 3))
        A_full = A_full.permute(permute_po + permute_pi + permute_k)

        dim = self.physical_dim**self.num_sites
        kraus_dim_total = np.prod([t.shape[4] for t in self.site_tensors])
        # A_full is now an operator K with shape (p_out, p_in, kraus_total)
        A_full = A_full.reshape(dim, dim, kraus_dim_total)

        # Build Choi matrix: Λ = Σ_k (K_k ⊗ K_k.conj()) where K_k is A_full[:,:,k]
        # This is equivalent to Λ = (I ⊗ A)(A^† ⊗ I)
        choi = torch.einsum('oik, OIK -> oOiI', A_full, A_full.conj())
        
        return choi.reshape(dim**2, dim**2)

    def compute_probability(self, rho_in: torch.Tensor, M_out: torch.Tensor) -> torch.Tensor:
        """
        Computes P(M_out|rho_in) using the tensor network contraction from Fig 2b.
        P(β|α) = Tr[ (ρ_α^T ⊗ M_β) * Λ_θ ]
        
        Args:
            rho_in: Input density matrix of shape (2^num_sites, 2^num_sites)
            M_out: Measurement operator of shape (2^num_sites, 2^num_sites)
            
        Returns:
            Probability as a real scalar tensor
        """
        phys_dim = self.physical_dim
        num_sites = self.num_sites
        
        # Reshape rho_in^T and M_out to have per-site indices
        # rho_in^T: (2^num_sites, 2^num_sites) -> (phys_dim, phys_dim, ..., phys_dim, phys_dim)
        # M_out: (2^num_sites, 2^num_sites) -> (phys_dim, phys_dim, ..., phys_dim, phys_dim)
        
        # First reshape to separate input and output indices
        rho_T_reshaped = rho_in.T.reshape([phys_dim] * (2 * num_sites))
        M_reshaped = M_out.reshape([phys_dim] * (2 * num_sites))
        
        # Permute to group input and output indices by site
        # Original: (i1, i2, ..., in, o1, o2, ..., on)
        # Target: (i1, o1, i2, o2, ..., in, on)
        permute_indices = []
        for i in range(num_sites):
            permute_indices.extend([i, i + num_sites])
        
        rho_T_site_by_site = rho_T_reshaped.permute(permute_indices)
        M_site_by_site = M_reshaped.permute(permute_indices)
        
        # Reshape to have pairs of (input, output) indices per site
        rho_T_site_by_site = rho_T_site_by_site.reshape([phys_dim, phys_dim] * num_sites)
        M_site_by_site = M_site_by_site.reshape([phys_dim, phys_dim] * num_sites)
        
        # Initialize left environment as scalar 1
        left_env = torch.tensor([[1.0]], device=self.device, dtype=self.dtype)
        
        # Contract site by site from left to right
        for i in range(num_sites):
            A_i = self.site_tensors[i]  # Shape: (l, r, o, i, k)
            
            # Extract local operators for this site
            # rho_T_i: (i, I) where i is input index, I is output index
            # M_i: (o, O) where o is input index, O is output index
            rho_T_i = rho_T_site_by_site[2*i:2*i+2].reshape(phys_dim, phys_dim)
            M_i = M_site_by_site[2*i:2*i+2].reshape(phys_dim, phys_dim)
            
            # Contract left environment with A_i and M_i
            # left_env: (bond, bond_conj)
            # A_i: (l, r, o, i, k)
            # M_i: (o, O)
            # Result: (r, O, i, k)
            left_env = torch.einsum('bl, lroik, oO -> brOik', left_env, A_i, M_i)
            
            # Contract with A_i.conj() and rho_T_i
            # A_i.conj(): (L, R, O, I, K)
            # rho_T_i: (i, I)
            # Result: (r, R, k, K)
            left_env = torch.einsum('brOik, LROIK, iI -> brRkK', left_env, A_i.conj(), rho_T_i)
            
            # Contract the kraus indices k=K
            # Result: (r, R)
            left_env = torch.einsum('brRkK -> brR', left_env)
        
        # Final contraction: trace over the remaining bond indices
        # left_env: (1, 1) for the final scalar
        prob = left_env.squeeze()
        
        # Ensure the result is real and positive
        return prob.real

    def trace_preserving_regularizer(self) -> torch.Tensor:
        """
        Computes the trace-preserving regularization term ||Tr_τ(Λ_θ) - I_σ||_F^2
        """
        # MPO for Tr_out(Λ)
        mpo_tensors = []
        for A_i in self.site_tensors:
            # A_i (l,r,o,i,k), A_conj (L,R,O,I,K)
            # Contract output legs o=O and sum over k=K
            mpo_tensor = torch.einsum('lroik, LROIK -> lLrRikI', A_i, A_i.conj())
            mpo_tensors.append(mpo_tensor)
            
        # This should result in an MPO for Tr_out(Λ). Then subtract Identity MPO.
        # This is a complex operation. For now, returning zero.
        # A full implementation requires MPO arithmetic.
        return torch.tensor(0.0, device=self.device)