# transformer.py — Transformer mini từ đầu
import math
import random

def dot_product(a, b):
    return sum(a[i]*b[i] for i in range(len(a)))

def mat_vec(W, x):
    return [sum(W[i][j]*x[j] for j in range(len(x)))
            for i in range(len(W))]

def softmax(x):
    e_x = [math.exp(i - max(x)) for i in x]
    tong = sum(e_x)
    return [i/tong for i in e_x]

def rmsnorm(x, weight):
    mean_sq = sum(i**2 for i in x) / len(x)
    rms = math.sqrt(mean_sq + 1e-5)
    return [x[i]/rms * weight[i] for i in range(len(x))]

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def silu(x):
    return x * sigmoid(x)

def addv(a, b):
    # Cộng 2 vector (residual connection)
    return [a[i]+b[i] for i in range(len(a))]


class Embedding:
    def __init__(self, vocab_size, embed_dim):
        random.seed(42)
        self.table = [
            [random.gauss(0, 0.02) for _ in range(embed_dim)]
            for _ in range(vocab_size)
        ]
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim

    def forward(self, token_id):
        return self.table[token_id]

class Attention:
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim
        random.seed(1)

        self.WQ = [[random.gauss(0,0.02) for _ in range(embed_dim)]
                   for _ in range(embed_dim)]
        self.WK = [[random.gauss(0,0.02) for _ in range(embed_dim)]
                   for _ in range(embed_dim)]
        self.WV = [[random.gauss(0,0.02) for _ in range(embed_dim)]
                   for _ in range(embed_dim)]
        self.WO = [[random.gauss(0,0.02) for _ in range(embed_dim)]
                   for _ in range(embed_dim)]

    def forward(self, xs):

        T   = len(xs)
        D   = self.embed_dim

        Qs = [mat_vec(self.WQ, x) for x in xs]
        Ks = [mat_vec(self.WK, x) for x in xs]
        Vs = [mat_vec(self.WV, x) for x in xs]

        out = []
        for t in range(T):

            scores = []
            for t2 in range(t+1):
                score = dot_product(Qs[t], Ks[t2]) / math.sqrt(D)
                scores.append(score)

            weights = softmax(scores)


            o = [sum(weights[t2] * Vs[t2][d] for t2 in range(t+1)) for d in range(D)]

            out.append(mat_vec(self.WO, o))

        return out

class FFN:
    def __init__(self, embed_dim, ffn_dim):
        random.seed(2)
        self.W1 = [[random.gauss(0,0.02) for _ in range(embed_dim)]
                   for _ in range(ffn_dim)]
        self.W2 = [[random.gauss(0,0.02) for _ in range(ffn_dim)]
                   for _ in range(embed_dim)]
        self.W3 = [[random.gauss(0,0.02) for _ in range(embed_dim)]
                   for _ in range(ffn_dim)]

    def forward(self, x):
        gate = [silu(v) for v in mat_vec(self.W1, x)]
        up   = mat_vec(self.W3, x)
        mid  = [gate[i]*up[i] for i in range(len(gate))]
        return mat_vec(self.W2, mid)

class TransformerBlock:
    def __init__(self, embed_dim, ffn_dim):
        self.attn    = Attention(embed_dim)
        self.ffn     = FFN(embed_dim, ffn_dim)
        self.norm1_w = [1.0] * embed_dim
        self.norm2_w = [1.0] * embed_dim

    def forward(self, xs):
        # Pre-norm + Attention + Residual
        normed = [rmsnorm(x, self.norm1_w) for x in xs]
        attn_out = self.attn.forward(normed)
        xs = [addv(xs[t], attn_out[t]) for t in range(len(xs))]

        xs = [addv(xs[t],
                   self.ffn.forward(rmsnorm(xs[t], self.norm2_w)))
              for t in range(len(xs))]
        return xs

class Transformer:
    def __init__(self, vocab_size, embed_dim, ffn_dim, n_layers):
        self.embed   = Embedding(vocab_size, embed_dim)
        self.blocks  = [TransformerBlock(embed_dim, ffn_dim)
                        for _ in range(n_layers)]
        self.norm_w  = [1.0] * embed_dim
        self.embed_dim   = embed_dim
        self.vocab_size  = vocab_size

    def forward(self, token_ids):
        xs = [self.embed.forward(tid) for tid in token_ids]
        for block in self.blocks:
            xs = block.forward(xs)
        last = rmsnorm(xs[-1], self.norm_w)
        logits = [dot_product(last, self.embed.table[v])
                  for v in range(self.vocab_size)]

        return logits

model = Transformer(
    vocab_size = 10,
    embed_dim  = 8,
    ffn_dim    = 16,
    n_layers   = 2
)

token_ids = [2, 5, 6, 4]
logits    = model.forward(token_ids)
probs     = softmax(logits)

print(f"Logits: {[round(l,3) for l in logits]}")
print(f"Probs tổng: {sum(probs):.4f}")  # phải = 1.0
print(f"Token chọn: {probs.index(max(probs))}")