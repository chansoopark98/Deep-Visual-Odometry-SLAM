import lietorch_backends
import torch
import torch.nn.functional as F


def _to_float32(inputs):
    """Convert floating point tensors to float32, keep others unchanged."""
    return tuple(x.float() if x.is_floating_point() else x for x in inputs)


class GroupOp(torch.autograd.Function):
    """ group operation base class """

    @classmethod
    def forward(cls, ctx, group_id, *inputs):
        ctx.group_id = group_id
        # Save original dtype for output conversion
        ctx.input_dtype = inputs[0].dtype
        # lietorch CUDA kernels require float32
        inputs_float = _to_float32(inputs)
        ctx.save_for_backward(*inputs_float)
        out = cls.forward_op(ctx.group_id, *inputs_float)
        # Restore original dtype
        return out.to(ctx.input_dtype)

    @classmethod
    def backward(cls, ctx, grad):
        error_str = "Backward operation not implemented for {}".format(cls)
        assert cls.backward_op is not None, error_str

        inputs = ctx.saved_tensors  # Already float32 from forward
        # Convert grad to float32 for CUDA kernel
        grad = grad.float().contiguous()
        grad_inputs = cls.backward_op(ctx.group_id, grad, *inputs)
        # Restore original dtype for gradients
        grad_inputs = tuple(g.to(ctx.input_dtype) if g.is_floating_point() else g for g in grad_inputs)
        return (None, ) + tuple(grad_inputs)
        

class Exp(GroupOp):
    """ exponential map """
    forward_op, backward_op = lietorch_backends.expm, lietorch_backends.expm_backward

class Log(GroupOp):
    """ logarithm map """
    forward_op, backward_op = lietorch_backends.logm, lietorch_backends.logm_backward

class Inv(GroupOp):
    """ group inverse """
    forward_op, backward_op = lietorch_backends.inv, lietorch_backends.inv_backward

class Mul(GroupOp):
    """ group multiplication """
    forward_op, backward_op = lietorch_backends.mul, lietorch_backends.mul_backward

class Adj(GroupOp):
    """ adjoint operator """
    forward_op, backward_op = lietorch_backends.adj, lietorch_backends.adj_backward

class AdjT(GroupOp):
    """ adjoint operator """
    forward_op, backward_op = lietorch_backends.adjT, lietorch_backends.adjT_backward

class Act3(GroupOp):
    """ action on point """
    forward_op, backward_op = lietorch_backends.act, lietorch_backends.act_backward

class Act4(GroupOp):
    """ action on point """
    forward_op, backward_op = lietorch_backends.act4, lietorch_backends.act4_backward

class Jinv(GroupOp):
    """ adjoint operator """
    forward_op, backward_op = lietorch_backends.Jinv, None

class ToMatrix(GroupOp):
    """ convert to matrix representation """
    forward_op, backward_op = lietorch_backends.as_matrix, None




### conversion operations to/from Euclidean embeddings ###

class FromVec(torch.autograd.Function):
    """ convert vector into group object """

    @classmethod
    def forward(cls, ctx, group_id, *inputs):
        ctx.group_id = group_id
        ctx.input_dtype = inputs[0].dtype
        inputs_float = _to_float32(inputs)
        ctx.save_for_backward(*inputs_float)
        return inputs[0]  # Pass through, keep original dtype

    @classmethod
    def backward(cls, ctx, grad):
        inputs = ctx.saved_tensors
        grad_float = grad.float()
        J = lietorch_backends.projector(ctx.group_id, *inputs)
        result = torch.matmul(grad_float.unsqueeze(-2), torch.linalg.pinv(J)).squeeze(-2)
        return None, result.to(ctx.input_dtype)

class ToVec(torch.autograd.Function):
    """ convert group object to vector """

    @classmethod
    def forward(cls, ctx, group_id, *inputs):
        ctx.group_id = group_id
        ctx.input_dtype = inputs[0].dtype
        inputs_float = _to_float32(inputs)
        ctx.save_for_backward(*inputs_float)
        return inputs[0]  # Pass through, keep original dtype

    @classmethod
    def backward(cls, ctx, grad):
        inputs = ctx.saved_tensors
        grad_float = grad.float()
        J = lietorch_backends.projector(ctx.group_id, *inputs)
        result = torch.matmul(grad_float.unsqueeze(-2), J).squeeze(-2)
        return None, result.to(ctx.input_dtype)

