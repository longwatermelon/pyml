from autograd import *
from math import sqrt, inf

# initialize [in_n, out_n] tensor
def tensor_init_rand(in_n, out_n):
    bound = np.sqrt(3./in_n)
    W = np.random.uniform(-bound, bound, size=(in_n, out_n))
    return W

class Module:
    def forward(self, X: Node) -> Node:
        raise NotImplementedError

    def parameters(self) -> list[Node]:
        return []

class Linear(Module):
    def __init__(self, in_n, out_n):
        self.W = Node.leaf(tensor_init_rand(in_n, out_n))
        self.b = Node.leaf(np.zeros((1, out_n)))

    def forward(self, X: Node) -> Node:
        wx = Node(Ftype.Matmul, [X, self.W])
        Z = Node(Ftype.Add, [wx, self.b])
        return Z

    def parameters(self) -> list[Node]:
        return [self.W, self.b]

class Relu(Module):
    def forward(self, X: Node) -> Node:
        return Node(Ftype.Relu, [X])

class LayerNorm(Module):
    def __init__(self, d):
        self.d = d
        self.gamma = Node.leaf(np.ones((d,)))
        self.beta = Node.leaf(np.zeros((d,)))
        self.eps = 1e-5

    # X: [T,d]
    def forward(self, X: Node) -> Node:
        # mean
        xsum = Node(Ftype.SumReduce, [X], axis=-1)
        ood = Node.leaf(np.asarray([1./self.d]))
        # [T,1]
        mu = Node(Ftype.Hadamard, [xsum, ood])

        # variance
        negative_mu = Node(Ftype.Negative, [mu])
        X_minus_mu = Node(Ftype.Add, [X, negative_mu])
        X_minus_mu_squared = Node(Ftype.Hadamard, [X_minus_mu, X_minus_mu])
        X_mms_sum = Node(Ftype.SumReduce, [X_minus_mu_squared], axis=-1)
        variance = Node(Ftype.Hadamard, [X_mms_sum, ood])

        # layernorm
        denom = Node(Ftype.Add, [variance, Node.leaf(np.asarray([self.eps]))])
        denom_sqrt = Node(Ftype.Sqrt, [denom])
        oo_denom_sqrt = Node(Ftype.Reciprocal, [denom_sqrt])
        frac = Node(Ftype.Hadamard, [X_minus_mu, oo_denom_sqrt])
        mul_gamma = Node(Ftype.Hadamard, [self.gamma, frac])
        plus_beta = Node(Ftype.Add, [mul_gamma, self.beta])

        return plus_beta

    def parameters(self) -> list[Node]:
        return [self.gamma, self.beta]

# Q : d -> d_k, K : d -> d_k, V : d -> d_v
def attention(Q: Node, K: Node, V: Node, mask=None) -> Node:
    KT = Node.transpose(K)
    QKT = Node(Ftype.Matmul, [Q, KT])

    d_k = Q.value.shape[-1]
    sqrt_dk = Node(Ftype.Leaf, [], value=np.asarray([sqrt(d_k)]))
    oo_sqrt_dk = Node(Ftype.Reciprocal, [sqrt_dk])
    raw_S = Node(Ftype.Hadamard, [QKT, oo_sqrt_dk])
    if mask is not None:
        raw_S = Node(Ftype.Add, [raw_S, mask])
    S = Node.softmax(raw_S, -1)

    return Node(Ftype.Matmul, [S,V])

class Attention(Module):
    def __init__(self, d, d_k, d_v):
        self.d = d
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = Node.leaf(tensor_init_rand(d, d_k))
        self.W_K = Node.leaf(tensor_init_rand(d, d_k))
        self.W_V = Node.leaf(tensor_init_rand(d, d_v))

    def forward(self, X: Node, mask=None) -> Node:
        Q = Node(Ftype.Matmul, [X, self.W_Q])
        K = Node(Ftype.Matmul, [X, self.W_K])
        V = Node(Ftype.Matmul, [X, self.W_V])

        return attention(Q,K,V, mask)

    def parameters(self) -> list[Node]:
        return [self.W_Q, self.W_K, self.W_V]

# d_k = d_v = d/h
class MHA(Module):
    def __init__(self, d, h, d_out):
        if d % h != 0:
            raise Exception("Need d % h == 0 for MHA")

        self.d = d
        self.h = h
        self.d_out = d_out

        # axis 0 is head index, axis 1,2 is its in/out
        self.W_Q = Node.leaf(np.stack([tensor_init_rand(d, d//h) for _ in range(h)], axis=0))
        self.W_K = Node.leaf(np.stack([tensor_init_rand(d, d//h) for _ in range(h)], axis=0))
        self.W_V = Node.leaf(np.stack([tensor_init_rand(d, d//h) for _ in range(h)], axis=0))
        # projects concat head outputs
        self.W_O = Node.leaf(tensor_init_rand(d, d_out))

    def forward(self, X: Node, mask=None) -> Node:
        Q = Node(Ftype.Matmul, [X, self.W_Q])
        K = Node(Ftype.Matmul, [X, self.W_K])
        V = Node(Ftype.Matmul, [X, self.W_V])

        # X comes in as [T,d], head_outs comes out as [h,T,d/h]
        head_outs = attention(Q,K,V, mask)
        permuted = Node(Ftype.Permute, [head_outs], permute_axes=(1,0,2))
        T = X.value.shape[0]
        head_concat = Node(Ftype.Reshape, [permuted], reshape_shape=(T, self.d))
        out = Node(Ftype.Matmul, [head_concat, self.W_O])

        return out

    def parameters(self) -> list[Node]:
        return [self.W_Q, self.W_K, self.W_V, self.W_O]

def create_causal_mask(T) -> np.ndarray:
    return np.triu(np.full((T,T), -inf), k=1)

class TfBlock(Module):
    def __init__(self, d, h):
        self.attn = MHA(d, h, d)
        self.mlp0 = Linear(d, 4*d)
        self.mlp1 = Linear(4*d, d)

        self.ln0 = LayerNorm(d)
        self.ln1 = LayerNorm(d)

    # X is [T,d]
    def forward(self, X: Node, mask=None) -> Node:
        # attn_out is [T,d]
        attn_out = self.attn.forward(X, mask)
        X = Node(Ftype.Add, [X, attn_out])
        X = self.ln0.forward(X)

        # mlp
        hidden_0 = Relu().forward(self.mlp0.forward(X))
        out = self.mlp1.forward(hidden_0)
        X = Node(Ftype.Add, [X, out])
        X = self.ln1.forward(X)

        return X

    def parameters(self) -> list[Node]:
        return [
            *self.attn.parameters(),
            *self.mlp0.parameters(),
            *self.mlp1.parameters(),
            *self.ln0.parameters(),
            *self.ln1.parameters()
        ]

class GPT(Module):
    def __init__(self, d, h, Tmax, N, V):
        self.d = d
        self.h = h
        self.Tmax = Tmax
        self.N = N
        self.V = V

        # embedding
        self.W_E = Node.leaf(np.random.uniform(low=-1., high=1., size=(V,d)))
        self.W_P = Node.leaf(np.random.uniform(low=-1., high=1., size=(Tmax,d)))

        # tf block chain
        self.blocks = list()
        for _ in range(N):
            self.blocks.append(TfBlock(d, h))

        # unembedding
        self.W_U = Node.leaf(tensor_init_rand(d, V))

    # X comes in as [T], token IDs in 0..V-1.
    def forward(self, X: Node) -> Node:
        T = X.value.shape[0]

        # embedding
        X_keepdim = np.reshape(X.value, (X.value.shape[0], 1))
        stream = Node(Ftype.Gather, [self.W_E], gather_inds=X_keepdim)

        # pos enc.
        pos_inds = np.arange(T, dtype=int).reshape((T,1))
        pos_info = Node(Ftype.Gather, [self.W_P], gather_inds=pos_inds)
        stream = Node(Ftype.Add, [stream, pos_info])

        # transformer block chain
        mask = Node.leaf(create_causal_mask(T))
        for block in self.blocks:
            stream = block.forward(stream, mask)

        # unembedding (shape [T,V])
        unembedded = Node(Ftype.Matmul, [stream, self.W_U])

        # softmax vocab logits
        scores = Node.softmax(unembedded, -1)

        return scores

    def parameters(self) -> list[Node]:
        params = [self.W_E, self.W_P, self.W_U]
        for block in self.blocks:
            params.extend(block.parameters())
        return params

