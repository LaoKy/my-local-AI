import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def silu(x):
    return x * sigmoid(x)
def softmax(x):
    e_x = [math.exp(i - max(x)) for i in x]
    tong = sum(e_x)
    return [j/tong for j in e_x]
def dot_product(a, b):
    return sum(a[i]*b[i] for i in range(len(a)))
def mat_vec(W, x):
    return [dot_product(W[i], x) for i in range(len(W))]

def silu_backward(z, dL_dout):
    dsilu_dz = sigmoid(z)+z*sigmoid(z)*(1-sigmoid(z))
    return dL_dout*dsilu_dz
def softmax_backward(p, dL_dp):
    dot = dot_product(p, dL_dp)
    return [p[i]*(dL_dp[i]-dot) for i in range(len(p))]
def softmax_crossentropy_backward(logits, targets):
    p = softmax(logits)
    return [p[i] - (1 if i == targets else 0) for i in range(len(p))]
def linear_backward(x, W, dL_dy):
    dL_dW = [[dL_dy[i]*x[j] for j in range(len(x))] for i in range(len(dL_dy))]
    dL_dx = [sum(dL_dy[i]*W[i][j] for i in range(len(dL_dy))) for j in range(len(W[0]))]
    return dL_dW, dL_dx

class AdamW:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, wd=0.01, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.wd = wd
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}
    def _step(self, w, g, m, v, i, j):
        if j is None:
            m[i] = self.beta1*m[i]+(1-self.beta1)*g
            v[i] = self.beta2*v[i]+(1-self.beta2)*g**2
            m_hat = m[i]/(1-self.beta1**self.t)
            v_hat = v[i]/(1-self.beta2**self.t)
        else:
            m[i][j] = self.beta1*m[i][j]+(1-self.beta1)*g
            v[i][j] = self.beta2*v[i][j]+(1-self.beta2)*g**2
            m_hat = m[i][j]/(1-self.beta1**self.t)
            v_hat = v[i][j]/(1-self.beta2**self.t)
        return w - self.lr*m_hat/(math.sqrt(v_hat)+self.eps) - (self.lr*self.wd*w)
    def update(self, name, W, dW):
        if name not in self.m:
            if isinstance(W[0], list):
                self.m[name] = [[0.0]*len(W[0]) for _ in range(len(W))]
                self.v[name] = [[0.0]*len(W[0]) for _ in range(len(W))]
            else:
                self.m[name] = [0.0]*len(W)
                self.v[name] = [0.0]*len(W)
        if isinstance(W[0], list):
            for i in range(len(W)):
                for j in range(len(W[0])):
                    W[i][j] = self._step(W[i][j], dW[i][j], self.m[name], self.v[name], i, j)
        else:
            for i in range(len(W)):
                W[i] = self._step(W[i], dW[i], self.m[name], self.v[name], i, None)

# ==================
# PHẦN 1: Model
# ==================
random.seed(42)
D_in, D_hid, D_out = 2, 4, 2

W1 = [[random.gauss(0,0.1) for _ in range(D_in)]
      for _ in range(D_hid)]
W2 = [[random.gauss(0,0.1) for _ in range(D_hid)]
      for _ in range(D_out)]

# ==================
# PHẦN 2: Forward
# ==================
def forward(x):
    h = [silu(sum(W1[i][j]*x[j] for j in range(D_in)))
         for i in range(D_hid)]
    logits = [sum(W2[i][j]*h[j] for j in range(D_hid))
              for i in range(D_out)]
    return logits, h

# ==================
# PHẦN 3: Train step (dùng AdamW)
# ==================
opt = AdamW(lr=0.01)

def train_step(x, target):
    # Forward
    logits, h = forward(x)

    # Loss
    probs = softmax(logits)
    print(target)
    loss  = -math.log(probs[target] + 1e-8)

    # Backward
    dL_dlogits = softmax_crossentropy_backward(logits, target)

    dL_dW2 = [[dL_dlogits[i]*h[j] for j in range(D_hid)]
              for i in range(D_out)]
    dL_dh  = [sum(dL_dlogits[i]*W2[i][j] for i in range(D_out))
              for j in range(D_hid)]

    z1     = [sum(W1[i][j]*x[j] for j in range(D_in))
              for i in range(D_hid)]
    dL_dz1 = [silu_backward(z1[i], dL_dh[i]) for i in range(D_hid)]
    dL_dW1 = [[dL_dz1[i]*x[j] for j in range(D_in)]
              for i in range(D_hid)]

    # Update bằng AdamW
    opt.t += 1
    opt.update('W1', W1, dL_dW1)
    opt.update('W2', W2, dL_dW2)

    return loss

# ==================
# PHẦN 4: Training loop
# ==================
dataset = [
    ([1.0, 0.0], 0),
    ([0.0, 1.0], 1),
]

print("Training với AdamW...")
for epoch in range(1, 201):
    total_loss = 0
    for x, target in dataset:
        total_loss += train_step(x, target)
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: loss={total_loss/len(dataset):.4f}")

# ==================
# PHẦN 5: Test
# ==================
print("\nTest:")
for x, target in dataset:
    logits, _ = forward(x)
    probs = softmax(logits)
    pred  = probs.index(max(probs))
    print(f"Input={x} → pred={pred} target={target} "
          f"{'✓' if pred==target else '✗'}")