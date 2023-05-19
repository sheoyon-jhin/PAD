import torch
import torchdiffeq
import torchsde
import warnings


def _check_compatability_per_tensor_base(control_gradient, z0):
    if control_gradient.shape[:-1] != z0.shape[:-1]:
        raise ValueError("X.derivative did not return a tensor with the same number of batch dimensions as z0. "
                         "X.derivative returned shape {} (meaning {} batch dimensions), whilst z0 has shape {} "
                         "(meaning {} batch dimensions)."
                         "".format(tuple(control_gradient.shape), tuple(control_gradient.shape[:-1]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))


def _check_compatability_per_tensor_forward(control_gradient, system, z0):
    _check_compatability_per_tensor_base(control_gradient, z0)
    if system.shape[:-2] != z0.shape[:-1]:
        raise ValueError("func did not return a tensor with the same number of batch dimensions as z0. func returned "
                         "shape {} (meaning {} batch dimensions), whilst z0 has shape {} (meaning {} batch"
                         " dimensions)."
                         "".format(tuple(system.shape), tuple(system.shape[:-2]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))
    if system.size(-2) != z0.size(-1):
        raise ValueError("func did not return a tensor with the same number of hidden channels as z0. func returned "
                         "shape {} (meaning {} channels), whilst z0 has shape {} (meaning {} channels)."
                         "".format(tuple(system.shape), system.size(-2), tuple(z0.shape), z0.size(-1)))
    if system.size(-1) != control_gradient.size(-1):
        raise ValueError("func did not return a tensor with the same number of input channels as X.derivative "
                         "returned. func returned shape {} (meaning {} channels), whilst X.derivative returned shape "
                         "{} (meaning {} channels)."
                         "".format(tuple(system.shape), system.size(-1), tuple(control_gradient.shape),
                                   control_gradient.size(-1)))


def _check_compatability_per_tensor_prod(control_gradient, vector_field, z0):
    _check_compatability_per_tensor_base(control_gradient, z0)
    if vector_field.shape != z0.shape:
        raise ValueError("func.prod did not return a tensor with the same shape as z0. func.prod returned shape {} "
                         "whilst z0 has shape {}."
                         "".format(tuple(vector_field.shape), tuple(z0.shape)))


def _check_compatability(X, func, z0, t):
    
    if not hasattr(X, 'derivative'):
        raise ValueError("X must have a 'derivative' method.")
    control_gradient = X.derivative(t[0].detach())
    if hasattr(func, 'prod'):
        is_prod = True
        vector_field = func.prod(t[0], z0, control_gradient)
    else:
        is_prod = False
        system = func(t[0], z0)

    if isinstance(z0, torch.Tensor):
        is_tensor = True
        if not isinstance(control_gradient, torch.Tensor):
            raise ValueError("z0 is a tensor and so X.derivative must return a tensor as well.")
        if is_prod:
            if not isinstance(vector_field, torch.Tensor):
                raise ValueError("z0 is a tensor and so func.prod must return a tensor as well.")
            _check_compatability_per_tensor_prod(control_gradient, vector_field, z0)
        else:
            if not isinstance(system, torch.Tensor):
                raise ValueError("z0 is a tensor and so func must return a tensor as well.")
            _check_compatability_per_tensor_forward(control_gradient, system, z0)

    elif isinstance(z0, (tuple, list)):
        is_tensor = False
        if not isinstance(control_gradient, (tuple, list)):
            raise ValueError("z0 is a tuple/list and so X.derivative must return a tuple/list as well.")
        if len(z0) != len(control_gradient):
            raise ValueError("z0 and X.derivative(t) must be tuples of the same length.")
        if is_prod:
            if not isinstance(vector_field, (tuple, list)):
                raise ValueError("z0 is a tuple/list and so func.prod must return a tuple/list as well.")
            if len(z0) != len(vector_field):
                raise ValueError("z0 and func.prod(t, z, dXdt) must be tuples of the same length.")
            for control_gradient_, vector_Field_, z0_ in zip(control_gradient, vector_field, z0):
                if not isinstance(control_gradient_, torch.Tensor):
                    raise ValueError("X.derivative must return a tensor or tuple of tensors.")
                if not isinstance(vector_Field_, torch.Tensor):
                    raise ValueError("func.prod must return a tensor or tuple/list of tensors.")
                _check_compatability_per_tensor_prod(control_gradient_, vector_Field_, z0_)
        else:
            if not isinstance(system, (tuple, list)):
                raise ValueError("z0 is a tuple/list and so func must return a tuple/list as well.")
            if len(z0) != len(system):
                raise ValueError("z0 and func(t, z) must be tuples of the same length.")
            for control_gradient_, system_, z0_ in zip(control_gradient, system, z0):
                if not isinstance(control_gradient_, torch.Tensor):
                    raise ValueError("X.derivative must return a tensor or tuple of tensors.")
                if not isinstance(system_, torch.Tensor):
                    raise ValueError("func must return a tensor or tuple/list of tensors.")
                _check_compatability_per_tensor_forward(control_gradient_, system_, z0_)

    else:
        raise ValueError("z0 must either a tensor or a tuple/list of tensors.")

    return is_tensor, is_prod


class _VectorField(torch.nn.Module):
    def __init__(self, X, func, is_tensor, is_prod):
        super(_VectorField, self).__init__()

        self.X = X
        self.func_s,self.func_c = func
        self.is_tensor = is_tensor
        self.is_prod = is_prod

        # torchsde backend
        self.sde_type = getattr(func, "sde_type", "stratonovich")
        self.noise_type = getattr(func, "noise_type", "additive")

    # torchdiffeq backend
    def forward(self, t, z):
        # control_gradient is of shape (..., input_channels)
        # import pdb;pdb.set_trace()
        control_gradient = self.X.derivative(t)
        vector_field = self.func_s(t, z)
        shared_field = self.func_c(t, z)
        
        out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)
        out2 = (shared_field @ control_gradient.unsqueeze(-1)).squeeze(-1)
        # import pdb ; pdb.set_trace()
        # out # + 0.0001*out2 
        return out 

    # def forward(self, t, z):
    #     # control_gradient is of shape (..., input_channels)
    #     control_gradient = self.X.derivative(t)
    #     import pdb ;pdb.set_trace()
    #     if self.is_prod:
    #         # out is of shape (..., hidden_channels)
    #         out = self.func.prod(t, z, control_gradient)
    #     else:
    #         # vector_field is of shape (..., hidden_channels, input_channels)
    #         vector_field,shared_field = self.func(t, z)
    #         if self.is_tensor:
    #             # out is of shape (..., hidden_channels)
    #             # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
    #             out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)
    #         else:
    #             out = tuple((vector_field_ @ control_gradient_.unsqueeze(-1)).squeeze(-1)
    #                         for vector_field_, control_gradient_ in zip(vector_field, control_gradient))
        
    #     return out

    # torchsde backend
    f = forward

    def g(self, t, z):
        return torch.zeros_like(z).unsqueeze(-1)


def cdeint(X, func, z0, t, adjoint=True, backend="torchdiffeq", **kwargs):
    if 'atol' not in kwargs:
        kwargs['atol'] = 1e-6
    if 'rtol' not in kwargs:
        kwargs['rtol'] = 1e-4
    if 'method' not in kwargs:
        kwargs['method'] = 'rk4'
    if kwargs['method'] == 'rk4':
        if 'options' not in kwargs:
            kwargs['options'] = {}
        options = kwargs['options']
        if 'step_size' not in options and 'grid_constructor' not in options:
            time_diffs = 0.5
            options['step_size'] = time_diffs
    if adjoint:
        if "adjoint_atol" not in kwargs:
            kwargs["adjoint_atol"] = kwargs["atol"]
        if "adjoint_rtol" not in kwargs:
            kwargs["adjoint_rtol"] = kwargs["rtol"]
    func_s,func_c = func
    is_tensor, is_prod = _check_compatability(X, func_s, z0, t)
    # is_tensor = True
    # is_prod=False
    
    
    vector_field = _VectorField(X=X, func=func, is_tensor=is_tensor, is_prod=is_prod)
    
    if backend == "torchdiffeq":
        odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
        
        out = odeint(func=vector_field, y0=z0, t=t, **kwargs)
    elif backend == "torchsde":
        sdeint = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint
        out = sdeint(sde=vector_field, y0=z0, ts=t, **kwargs)
    else:
        raise ValueError(f"Unrecognised backend={backend}")

    if is_tensor:
        batch_dims = range(1, len(out.shape) - 1)
        out = out.permute(*batch_dims, 0, -1)
    else:
        out_ = []
        for outi in out:
            batch_dims = range(1, len(outi.shape) - 1)
            outi = outi.permute(*batch_dims, 0, -1)
            out_.append(outi)
        out = tuple(out_)

    return out 



class _VectorField_Anomaly(torch.nn.Module):
    def __init__(self, X, func, is_tensor, is_prod):
        super(_VectorField_Anomaly, self).__init__()

        self.X = X
        self.func = func
        self.is_tensor = is_tensor
        self.is_prod = is_prod

        # torchsde backend
        self.sde_type = getattr(func, "sde_type", "stratonovich")
        self.noise_type = getattr(func, "noise_type", "additive")

    # torchdiffeq backend
    def forward(self, t, z):
        # control_gradient is of shape (..., input_channels)
        # import pdb;pdb.set_trace()
        control_gradient = self.X.derivative(t)
        vector_field = self.func(t, z)
        # shared_field = self.func_c(t, z)
        
        out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)
        # out2 = (shared_field @ control_gradient.unsqueeze(-1)).squeeze(-1)
        # # import pdb ; pdb.set_trace()
        # # out # + 0.0001*out2 
        return out 

    
    f = forward

    def g(self, t, z):
        return torch.zeros_like(z).unsqueeze(-1)
class _VectorField(torch.nn.Module):
    def __init__(self, X, func, is_tensor, is_prod):
        super(_VectorField, self).__init__()

        self.X = X
        self.func_s,self.func_c = func
        self.is_tensor = is_tensor
        self.is_prod = is_prod

        # torchsde backend
        self.sde_type = getattr(func, "sde_type", "stratonovich")
        self.noise_type = getattr(func, "noise_type", "additive")

    # torchdiffeq backend
    def forward(self, t, z):
        # control_gradient is of shape (..., input_channels)
        # import pdb;pdb.set_trace()
        control_gradient = self.X.derivative(t)
        vector_field = self.func_s(t, z)
        shared_field = self.func_c(t, z)
        
        out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)
        out2 = (shared_field @ control_gradient.unsqueeze(-1)).squeeze(-1)
        # import pdb ; pdb.set_trace()
        # out # + 0.0001*out2 
        return out 

    
    f = forward

    def g(self, t, z):
        return torch.zeros_like(z).unsqueeze(-1)


def cdeint_anomaly(X, func, z0, t, adjoint=True, backend="torchdiffeq", **kwargs):
    if 'atol' not in kwargs:
        kwargs['atol'] = 1e-6
    if 'rtol' not in kwargs:
        kwargs['rtol'] = 1e-4
    if 'method' not in kwargs:
        kwargs['method'] = 'rk4'
    if kwargs['method'] == 'rk4':
        if 'options' not in kwargs:
            kwargs['options'] = {}
        options = kwargs['options']
        if 'step_size' not in options and 'grid_constructor' not in options:
            time_diffs = 0.5
            options['step_size'] = time_diffs
    if adjoint:
        if "adjoint_atol" not in kwargs:
            kwargs["adjoint_atol"] = kwargs["atol"]
        if "adjoint_rtol" not in kwargs:
            kwargs["adjoint_rtol"] = kwargs["rtol"]
    # func_s,func_c = func
    is_tensor, is_prod = _check_compatability(X, func, z0, t)
    # is_tensor = True
    # is_prod=False
    
    
    vector_field = _VectorField_Anomaly(X=X, func=func, is_tensor=is_tensor, is_prod=is_prod)
    
    if backend == "torchdiffeq":
        odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
        
        out = odeint(func=vector_field, y0=z0, t=t, **kwargs)
    elif backend == "torchsde":
        sdeint = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint
        out = sdeint(sde=vector_field, y0=z0, ts=t, **kwargs)
    else:
        raise ValueError(f"Unrecognised backend={backend}")

    if is_tensor:
        batch_dims = range(1, len(out.shape) - 1)
        out = out.permute(*batch_dims, 0, -1)
    else:
        out_ = []
        for outi in out:
            batch_dims = range(1, len(outi.shape) - 1)
            outi = outi.permute(*batch_dims, 0, -1)
            out_.append(outi)
        out = tuple(out_)

    return out 







