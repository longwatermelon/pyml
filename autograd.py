import numpy as np
from enum import Enum

class Ftype(Enum):
    Matmul = 1
    Add = 2
    Hadamard = 3
    Relu = 4
    Exp = 5
    Log = 6
    SumReduce = 7
    MaxReduce = 8
    Leaf = 9
    Permute = 10
    Reciprocal = 11
    Negative = 12
    Reshape = 13
    Gather = 14
    Sqrt = 15

    # run forward pass of function, return result
    def forward(self, args, *, axis=None, permute_axes=None, reshape_shape=None, gather_inds=None) -> np.ndarray:
        if self == Ftype.Matmul:
            if args[0].value.ndim < 2 or args[1].value.ndim < 2:
                raise ValueError("Matmul requires args with >= 2 dims.")

            return args[0].value @ args[1].value
        if self == Ftype.Add:
            return args[0].value + args[1].value
        if self == Ftype.Hadamard:
            return args[0].value * args[1].value
        if self == Ftype.Relu:
            return np.maximum(args[0].value, 0)
        if self == Ftype.Exp:
            return np.exp(args[0].value)
        if self == Ftype.Log:
            return np.log(args[0].value)
        if self == Ftype.SumReduce:
            if axis == None:
                raise ValueError("axis must not be None for SumReduce forward")

            return np.sum(args[0].value, axis=axis, keepdims=True)
        if self == Ftype.MaxReduce:
            if axis == None:
                raise ValueError("axis must not be None for MaxReduce forward")

            return np.max(args[0].value, axis=axis, keepdims=True)
        if self == Ftype.Permute:
            if permute_axes == None:
                raise ValueError("permute_axes must not be None for Permute forward")

            return np.transpose(args[0].value, permute_axes)
        if self == Ftype.Reciprocal:
            return 1. / args[0].value
        if self == Ftype.Negative:
            return -args[0].value
        if self == Ftype.Reshape:
            if reshape_shape == None:
                raise ValueError("reshape_shape must not be None for Reshape forward")

            return np.reshape(args[0].value, reshape_shape)
        if self == Ftype.Gather:
            if gather_inds is None:
                raise ValueError("gather_inds must not be None for Gather forward")

            return args[0].value[tuple(np.moveaxis(gather_inds, -1, 0))]
        if self == Ftype.Sqrt:
            return np.sqrt(args[0].value)

        raise Exception()

    # run backward pass of a function, return gradient wrt ith argument
    # G = ∂X/∂(function output), so ∂X/∂args[i] = G ∂(function output)/∂args[i]
    def backward(self, args, i, G, *, axis=None, permute_axes=None, gather_inds=None) -> np.ndarray:
        out = np.ndarray([])

        if self == Ftype.Matmul:
            if i == 0:
                out = G @ args[1].value.mT
            if i == 1:
                out = args[0].value.mT @ G

        if self == Ftype.Add:
            out = G

        if self == Ftype.Hadamard:
            if i == 0:
                out = G * args[1].value
            if i == 1:
                out = G * args[0].value

        if self == Ftype.Relu:
            out = G * (args[0].value > 0).astype(int)

        if self == Ftype.Exp:
            out = G * np.exp(args[0].value)

        if self == Ftype.Log:
            out = G * 1./args[0].value

        if self == Ftype.SumReduce:
            out = np.broadcast_to(G, args[0].value.shape)

        if self == Ftype.MaxReduce:
            if axis == None:
                raise ValueError("axis must not be None for MaxReduce.")

            # max mask
            inds = np.argmax(args[0].value, axis=axis, keepdims=True)
            mask = np.zeros(args[0].value.shape, dtype=int)
            np.put_along_axis(mask, inds, 1, axis=axis)
            out = G * mask

        if self == Ftype.Permute:
            if permute_axes == None:
                raise ValueError("permute_axes must not be None for Permute.")

            # each A[i] for index tuple i gets sent to fout[i'] where i' = permute(i, permute\_axes), then dX/dA[i] = dX/dfout[i'] dfout[i']/dA[i]... since fout[i'] = A[i], dfout[i']/dA[i] = 1, so dX/dA[i] = dX/dfout[i']. So you basically un-permute G via permute\_axes.
            out = np.transpose(G, np.argsort(permute_axes))

        if self == Ftype.Gather:
            pass

        if self == Ftype.Reciprocal:
            out = G * (-1. / (args[0].value * args[0].value))

        if self == Ftype.Negative:
            out = -G

        if self == Ftype.Reshape:
            out = G.reshape(args[0].value.shape)

        if self == Ftype.Gather:
            if gather_inds is None:
                raise ValueError("gather_inds must not be None for Gather.")

            out = np.zeros(args[0].value.shape)
            fout_shape = gather_inds.shape[:-1]
            for dst_ind in np.ndindex(fout_shape):
                src_ind = tuple(gather_inds[dst_ind])
                out[src_ind] += G[dst_ind]

        if self == Ftype.Sqrt:
            out = G * 1. / (2. * np.sqrt(args[0].value))

        # need to unbroadcast to output shape: collect contributions for each entry
        # across all entries it was broadcasted to
        target_shape = args[i].value.shape
        out_shape = out.shape

        # delete extra axes
        extra_axes = len(out_shape) - len(target_shape)
        for _ in range(0, extra_axes):
            out = np.sum(out, axis=0, keepdims=False)

        # unbroadcast non-1 axes for 1 in target shape
        for j in range(0, len(target_shape)):
            if args[i].value.shape[j] == 1 and out.shape[j] > 1:
                out = np.sum(out, axis=j, keepdims=True)

        return out

class Node:
    def __init__(self, ftype: Ftype, args: list["Node"], *, axis=None, permute_axes=None, reshape_shape=None, gather_inds=None, value=None):
        self.ftype = ftype
        self.args = args

        if value is not None:
            self.value = value
        else:
            self.value = ftype.forward(args, axis=axis,
                                             permute_axes=permute_axes,
                                             reshape_shape=reshape_shape,
                                             gather_inds=gather_inds)

        self.axis = axis
        self.permute_axes = permute_axes
        self.reshape_shape = reshape_shape
        self.gather_inds = gather_inds
        self.zero_grad()

    @classmethod
    def leaf(cls, value):
        return cls(Ftype.Leaf, [], value=value)

    @classmethod
    def transpose(cls, node):
        n = len(node.value.shape)
        return cls(Ftype.Permute, [node], permute_axes=(*range(n-2), n-1, n-2))

    @classmethod
    def softmax(cls, node, axis):
        max_axis = cls.leaf(np.max(node.value, axis=axis, keepdims=True))
        neg_max_axis = cls(Ftype.Negative, [max_axis])
        shifted = cls(Ftype.Add, [node, neg_max_axis])
        exp_node = cls(Ftype.Exp, [shifted])

        sum_exp = cls(Ftype.SumReduce, [exp_node], axis=axis)
        oo_sum_exp = cls(Ftype.Reciprocal, [sum_exp])
        weights = cls(Ftype.Hadamard, [exp_node, oo_sum_exp])

        return weights

    # assume self.grad is correct, contribute to arg.grad for each arg in self.args
    def add_child_grads(self):
        for i in range(len(self.args)):
            self.args[i].grad += self.ftype.backward(self.args, i, self.grad,
                                                     axis=self.axis,
                                                     permute_axes=self.permute_axes,
                                                     gather_inds=self.gather_inds)

    # set self.grad = 0
    def zero_grad(self):
        self.grad = np.full(self.value.shape, 0, dtype=float)

# populate .grad variables for all nodes in autograd graph
def autograd(root: Node):
    # topo order
    seen = set()
    order = list()
    def dfs(u: Node):
        if u in seen:
            return

        seen.add(u)
        for v in u.args:
            dfs(v)

        order.append(u)

    dfs(root)
    order = order[::-1]

    # run gradient flows through topo order
    for u in order:
        u.zero_grad()

    root.grad = np.full(root.value.shape, 1, dtype=float)
    for u in order:
        u.add_child_grads()

