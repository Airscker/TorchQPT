import torch
import numpy as np
from typing import List, Optional, Union
from ._base import BaseModel, COMPLEX_DTYPE

class LPDO(BaseModel):
    """
    Represents a Locally-Purified Density Operator (LPDO) for quantum process tomography.
    
    The LPDO is used to represent the Choi matrix of a quantum channel using a tensor network
    structure. Each site tensor has physical indices (sigma, tau), bond indices (mu), and
    a Kraus index (nu) for representing mixed states.
    
    Attributes:
        site_tensors (List[torch.Tensor]): List of tensors defining the LPDO.
        num_sites (int): Number of sites (qubits) in the system.
        physical_dims (List[int]): Physical dimensions for each site.
        bond_dims (List[int]): Bond dimensions between sites.
        kraus_dims (List[int]): Kraus dimensions for each site.
        _device (torch.device): PyTorch device of the tensors.
        _dtype (torch.dtype): PyTorch dtype of the tensors.
    """
    def __init__(self, 
                 site_tensors: List[torch.Tensor],
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: Optional[torch.dtype] = None,
                 **kwargs):
        """
        Initialize an LPDO.
        
        Args:
            site_tensors (List[torch.Tensor]): List of tensors defining the LPDO.
                Each tensor should have shape:
                - Site 0: (physical_dim, kraus_dim, right_bond_dim)
                - Middle sites: (left_bond_dim, physical_dim, kraus_dim, right_bond_dim)
                - Site N-1: (left_bond_dim, physical_dim, kraus_dim)
            device (Optional[Union[str, torch.device]]): Target device for tensors.
            dtype (Optional[torch.dtype]): Target dtype for tensors.
            **kwargs: Additional arguments passed to BaseModel.
        """
        super().__init__(site_tensors=site_tensors, device=device, dtype=dtype, **kwargs)
        
        if not site_tensors:
            raise ValueError("site_tensors list cannot be empty.")
            
        self.num_sites = len(site_tensors)
        
        # Set device and dtype
        self._device = torch.device(device) if device is not None else site_tensors[0].device
        self._dtype = dtype if dtype is not None else site_tensors[0].dtype
        
        # Check consistency of devices and dtypes
        devices = {t.device for t in site_tensors}
        dtypes = {t.dtype for t in site_tensors}
        if len(devices) > 1 or len(dtypes) > 1:
            raise ValueError(
                f"All site_tensors must have the same device and dtype. "
                f"Found devices: {devices}, dtypes: {dtypes}"
            )
            
        # Extract dimensions
        self.physical_dims = []
        self.bond_dims = []
        self.kraus_dims = []
        
        # Register site tensors as parameters
        self.site_tensors = torch.nn.ParameterList()
        for i, t in enumerate(site_tensors):
            # Convert tensor to parameter and register it
            param = torch.nn.Parameter(t)
            self.site_tensors.append(param)
            
            if i == 0:  # Leftmost site
                if t.ndim != 3:
                    raise ValueError(f"Tensor at site 0 must be rank 3 (phys, kraus, D_0), got rank {t.ndim}")
                self.physical_dims.append(t.shape[0])
                self.kraus_dims.append(t.shape[1])
                if self.num_sites > 1:
                    self.bond_dims.append(t.shape[2])
            elif i == self.num_sites - 1:  # Rightmost site
                if t.ndim != 3:
                    raise ValueError(f"Tensor at site {i} must be rank 3 (D_{i-1}, phys, kraus), got rank {t.ndim}")
                self.physical_dims.append(t.shape[1])
                self.kraus_dims.append(t.shape[2])
                if self.num_sites > 1 and t.shape[0] != self.bond_dims[-1]:
                    raise ValueError(
                        f"Left bond dimension mismatch at site {i}: tensor shape {t.shape[0]} vs previous right bond {self.bond_dims[-1]}"
                    )
            else:  # Middle sites
                if t.ndim != 4:
                    raise ValueError(f"Tensor at site {i} must be rank 4 (D_{i-1}, phys, kraus, D_i), got rank {t.ndim}")
                self.physical_dims.append(t.shape[1])
                self.kraus_dims.append(t.shape[2])
                if t.shape[0] != self.bond_dims[-1]:
                    raise ValueError(
                        f"Left bond dimension mismatch at site {i}: tensor shape {t.shape[0]} vs previous right bond {self.bond_dims[-1]}"
                    )
                self.bond_dims.append(t.shape[3])
    
    @property
    def device(self) -> torch.device:
        """The PyTorch device on which the LPDO tensors are stored."""
        return self._device
    
    @property
    def dtype(self) -> torch.dtype:
        """The PyTorch dtype of the LPDO tensors."""
        return self._dtype
    
    @classmethod
    def random_initialization(cls,
                            num_sites: int,
                            physical_dim: int = 2,
                            bond_dim: int = 2,
                            kraus_dim: int = 1,
                            device: Optional[Union[str, torch.device]] = None,
                            dtype: Optional[torch.dtype] = None) -> 'LPDO':
        """
        Creates a randomly initialized LPDO.
        
        Args:
            num_sites (int): Number of sites (qubits) in the system.
            physical_dim (int): Physical dimension of each site (default: 2 for qubits).
            bond_dim (int): Bond dimension between sites.
            kraus_dim (int): Kraus dimension for each site.
            device (Optional[Union[str, torch.device]]): Target device for tensors.
            dtype (Optional[torch.dtype]): Target dtype for tensors.
            
        Returns:
            LPDO: A randomly initialized LPDO.
        """
        dev = torch.device(device) if device is not None else torch.device('cpu')
        _dtype = dtype if dtype is not None else COMPLEX_DTYPE
        
        site_tensors = []
        for i in range(num_sites):
            if i == 0:  # Leftmost site
                shape = (physical_dim, kraus_dim, bond_dim)
            elif i == num_sites - 1:  # Rightmost site
                shape = (bond_dim, physical_dim, kraus_dim)
            else:  # Middle sites
                shape = (bond_dim, physical_dim, kraus_dim, bond_dim)
                
            # Initialize with random complex values
            tensor = torch.randn(shape, device=dev, dtype=_dtype)
            # Normalize to help with initial stability
            tensor = tensor / torch.norm(tensor)
            site_tensors.append(tensor)
            
        return cls(site_tensors, device=dev, dtype=_dtype)
    
    def compute_probability(self, input_state: torch.Tensor, measurement: torch.Tensor) -> torch.Tensor:
        """
        Computes the probability of measuring the given output state for a given input state.
        
        This implements the formula from the paper:
        P(beta|alpha) = Tr(M_beta @ E(rho_alpha))
        where E is represented by our LPDO.
        
        The tensor network contraction is performed as follows:
        1. Contract input state with first site tensor
        2. Contract result with middle site tensors
        3. Contract with last site tensor
        4. Contract with measurement operator
        5. Take the trace to get the probability
        
        Args:
            input_state (torch.Tensor): Input state tensor of shape (2^num_qubits, 2^num_qubits).
            measurement (torch.Tensor): Measurement operator tensor of shape (2^num_qubits, 2^num_qubits).
            
        Returns:
            torch.Tensor: Probability of the measurement outcome.
        """
        # Reshape input state and measurement to match tensor network structure
        # For a 2-qubit system, reshape from (4,4) to (2,2,2,2)
        input_reshaped = input_state.reshape([2] * (2 * self.num_sites))
        measurement_reshaped = measurement.reshape([2] * (2 * self.num_sites))
        
        # Contract with first site tensor
        # First site tensor shape: (physical_dim, kraus_dim, bond_dim)
        # Input shape: (2,2,2,2) for 2 qubits
        # Contract physical indices
        result = torch.tensordot(self.site_tensors[0], input_reshaped, dims=([0], [0]))
        
        # Contract with middle site tensors
        for i in range(1, self.num_sites - 1):
            # Middle site tensor shape: (bond_dim, physical_dim, kraus_dim, bond_dim)
            # Contract physical and bond indices
            result = torch.tensordot(result, self.site_tensors[i], dims=([-1], [0]))
            result = torch.tensordot(result, input_reshaped, dims=([1], [i]))
        
        # Contract with last site tensor
        # Last site tensor shape: (bond_dim, physical_dim, kraus_dim)
        result = torch.tensordot(result, self.site_tensors[-1], dims=([-1], [0]))
        result = torch.tensordot(result, input_reshaped, dims=([1], [-1]))
        
        # Contract with measurement operator
        # Reshape measurement to match remaining indices
        measurement_reshaped = measurement.reshape([2] * (2 * self.num_sites))
        result = torch.tensordot(result, measurement_reshaped, dims=([0, 1], [0, 1]))
        
        # The result should now be a scalar
        prob = result.real
        
        # Ensure the probability is non-negative
        return torch.abs(prob)
    
    def trace_preserving_regularizer(self) -> torch.Tensor:
        """
        Computes the trace-preserving regularization term.
        
        This implements the regularization term from the paper:
        ||Tr_tau(Lambda_theta) - I_sigma||_F^2
        
        The regularization ensures that the quantum channel preserves the trace of input states.
        
        Returns:
            torch.Tensor: The regularization term measuring deviation from trace preservation.
        """
        # Initialize the partial trace
        partial_trace = torch.eye(2**self.num_sites, device=self.device, dtype=self.dtype)
        
        # Contract the LPDO tensors to compute the partial trace
        for i, tensor in enumerate(self.site_tensors):
            if i == 0:
                # First site: contract physical and Kraus indices
                partial_trace = torch.tensordot(tensor, tensor.conj(), dims=([0, 1], [0, 1]))
            elif i == self.num_sites - 1:
                # Last site: contract physical and Kraus indices
                partial_trace = torch.tensordot(partial_trace, tensor, dims=([-1], [0]))
                partial_trace = torch.tensordot(partial_trace, tensor.conj(), dims=([-1], [0]))
            else:
                # Middle sites: contract physical and Kraus indices
                partial_trace = torch.tensordot(partial_trace, tensor, dims=([-1], [0]))
                partial_trace = torch.tensordot(partial_trace, tensor.conj(), dims=([-1], [0]))
        
        # Compute the difference from identity
        identity = torch.eye(2**self.num_sites, device=self.device, dtype=self.dtype)
        diff = partial_trace - identity
        
        # Return the Frobenius norm squared
        return torch.norm(diff, p='fro')**2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LPDO model.
        
        This applies the quantum channel represented by the LPDO to an input state.
        
        Args:
            x (torch.Tensor): Input tensor representing a quantum state of shape (2^num_qubits, 2^num_qubits).
            
        Returns:
            torch.Tensor: Output tensor after applying the quantum channel.
        """
        # Reshape input to match tensor network structure
        x_reshaped = x.reshape([2] * (2 * self.num_sites))
        
        # Contract with first site tensor
        result = torch.tensordot(self.site_tensors[0], x_reshaped, dims=([0], [0]))
        
        # Contract with middle site tensors
        for i in range(1, self.num_sites - 1):
            result = torch.tensordot(result, self.site_tensors[i], dims=([-1], [0]))
            result = torch.tensordot(result, x_reshaped, dims=([1], [i]))
        
        # Contract with last site tensor
        result = torch.tensordot(result, self.site_tensors[-1], dims=([-1], [0]))
        result = torch.tensordot(result, x_reshaped, dims=([1], [-1]))
        
        # Reshape back to matrix form
        output = result.reshape(2**self.num_sites, 2**self.num_sites)
        
        return output
    
    def __repr__(self) -> str:
        """String representation of the LPDO."""
        _str = super().__repr__()
        _str += f"num_sites={self.num_sites}, "
        _str += f"physical_dims={self.physical_dims}, "
        _str += f"bond_dims={self.bond_dims}, "
        _str += f"kraus_dims={self.kraus_dims})"
        return _str
    
    def __len__(self) -> int:
        """Number of sites in the LPDO."""
        return self.num_sites
