import math

def dot_product(a, b):
    return sum(a[i]*b[i] for i in range(len(a)))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def silu(x):
    return x * sigmoid(x)

def softmax(x):
    e_x = [math.exp(i - max(x)) for i in x]
    tong = sum(e_x)
    return [i/tong for i in e_x]

def mat_vec(W, x):
    return [sum(W[i][j]*x[j] for j in range(len(x)))
            for i in range(len(W))]

def linear_backward(W, x, dL_dy):
    out_dim = len(dL_dy)
    in_dim  = len(x)
    dL_dW = [[dL_dy[i] * x[j] for j in range(in_dim)] for i in range(out_dim)]
    dL_dx = [sum(dL_dy[i] * W[i][j] for i in range(out_dim)) for j in range(in_dim)]
    return dL_dW, dL_dx

def silu_backward(z, dL_dout):
    sig      = sigmoid(z)
    dsilu_dz = sig + z*(sig*(1-sig))
    return dL_dout * dsilu_dz

def ffn_forward(x, W1, W2, W3):
    z1   = mat_vec(W1, x)
    gate = [silu(v) for v in z1]
    up   = mat_vec(W3, x)
    mid  = [gate[i]*up[i] for i in range(len(gate))]
    out  = mat_vec(W2, mid)
    cache = (x, z1, gate, up, mid)
    return out, cache

def ffn_backward(dL_dout, cache, W1, W2, W3):
    x, z1, gate, up, mid = cache
    dL_dW2, dL_dmid = linear_backward(W2, mid, dL_dout)
    dL_dgate = [dL_dmid[i] * up[i]   for i in range(len(mid))]
    dL_dup   = [dL_dmid[i] * gate[i] for i in range(len(mid))]
    dL_dz1 = [silu_backward(z1[i], dL_dgate[i]) for i in range(len(z1))]
    dL_dW1, dL_dx_from_W1 = linear_backward(W1, x, dL_dz1)
    dL_dW3, dL_dx_from_W3 = linear_backward(W3, x, dL_dup)
    dL_dx = [dL_dx_from_W1[i] + dL_dx_from_W3[i]for i in range(len(x))]
    return dL_dx, dL_dW1, dL_dW2, dL_dW3

def attention_backward_step3(dL_dresult, outs, WO):
    T = len(outs)
    D = len(outs[0])

    dL_douts = []
    dL_dWO = [[0.0]*D for _ in range(D)]

    for t in range(T):
        dL_dWO_t, dL_dout_t = linear_backward(WO, outs[t], dL_dresult[t])
        for i in range(D):
            for j in range(D):
                dL_dWO[i][j] += dL_dWO_t[i][j] 
        dL_douts.append(dL_dout_t)
    return dL_douts, dL_dWO

def attention_backward_step2(dL_douts, weights_list, Vs, T, D):
    dL_dVs = [[0.0]*D for _ in range(T)]
    dL_dweights_list = []

    for t in range(T):
        weights  = weights_list[t]
        dL_dw    = [0.0] * (t+1)
        for s in range(t+1):
            for d in range(D):
                dL_dVs[s][d] += dL_douts[t][d] * weights[s]
                dL_dw[s] += dL_douts[t][d] * Vs[s][d]
        dL_dweights_list.append(dL_dw)
    return dL_dweights_list, dL_dVs

def softmax_backward(p, dL_dp):
    dot = sum(dL_dp[j]*p[j] for j in range(len(dL_dp)))
    dL_dscores = [p[i] * (dL_dp[i] - dot) for i in range(len(p))]
    return dL_dscores

def scores_backward(dL_dscores_t, Qs_t, Ks, t, D):
    scale    = math.sqrt(D)
    dL_dQs_t = [0.0] * D
    dL_dKs   = [[0.0]*D for _ in range(t+1)]
    for s in range(t+1):
        for d in range(D):
            dL_dQs_t[d] += dL_dscores_t[s] * Ks[s][d] / scale
            dL_dKs[s][d] += dL_dscores_t[s] * Qs_t[d] / scale
    return dL_dQs_t, dL_dKs

def attention_backward(dL_dresult, cache, WQ, WK, WV, WO):
    xs, Qs, Ks, Vs, weights_list, outs_list = cache
    T = len(xs)
    D = len(xs[0])

    dL_dWQ = [[0.0]*D for _ in range(D)]
    dL_dWK = [[0.0]*D for _ in range(D)]
    dL_dWV = [[0.0]*D for _ in range(D)]
    dL_dWO = [[0.0]*D for _ in range(D)]
    dL_dxs = [[0.0]*D for _ in range(T)]

    dL_douts, dL_dWO = attention_backward_step3(dL_dresult, outs_list, WO)
    dL_dweights_list, dL_dVs = attention_backward_step2(dL_douts, weights_list, Vs, T, D)
    dL_dQs = [[0.0]*D for _ in range(T)]
    dL_dKs = [[0.0]*D for _ in range(T)]

    for t in range(T):
        dL_dscores_t = softmax_backward(weights_list[t], dL_dweights_list[t])
        dL_dQs_t, dL_dKs_t = scores_backward(dL_dscores_t, Qs[t], Ks, t, D)
        for d in range(D):
            dL_dQs[t][d] += dL_dQs_t[d]
        for s in range(t+1):
            for d in range(D):
                dL_dKs[s][d] += dL_dKs_t[s][d]

    for t in range(T):
        dL_dWQ_t, dL_dx_Q = linear_backward(WQ, xs[t], dL_dQs[t])
        dL_dWK_t, dL_dx_K = linear_backward(WK, xs[t], dL_dKs[t])
        dL_dWV_t, dL_dx_V = linear_backward(WV, xs[t], dL_dVs[t])
        for i in range(D):
            for j in range(D):
                dL_dWQ[i][j] += dL_dWQ_t[i][j]
                dL_dWK[i][j] += dL_dWK_t[i][j]
                dL_dWV[i][j] += dL_dWV_t[i][j]
        for d in range(D):
            dL_dxs[t][d] += dL_dx_Q[d] + dL_dx_K[d] + dL_dx_V[d]
    return dL_dxs, dL_dWQ, dL_dWK, dL_dWV, dL_dWO

class AdamW:
    def __init__(self, lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.wd    = wd
        self.t     = 0
        self.ms    = {}
        self.vs    = {}

    def step(self, params, grads):
        self.t += 1

        for name in params:
            if name not in self.ms:
                self.ms[name] = [[0.0]*len(params[name][0])
                                 for _ in range(len(params[name]))] \
                                if isinstance(params[name][0], list) \
                                else [0.0]*len(params[name])
                self.vs[name] = [[0.0]*len(params[name][0])
                                 for _ in range(len(params[name]))] \
                                if isinstance(params[name][0], list) \
                                else [0.0]*len(params[name])

            self._update(params[name], grads[name],
                        self.ms[name], self.vs[name])

    def _update_element(self, w, g, m, v):
        m_new = self.beta1 * m + (1-self.beta1) * g
        v_new = self.beta2 * v + (1-self.beta2) * g**2
        m_hat = m_new / (1 - self.beta1**self.t)
        v_hat = v_new / (1 - self.beta2**self.t)
        w_new = w - self.lr * m_hat / (v_hat**0.5 + self.eps) \
                  - self.lr * self.wd * w
        return w_new, m_new, v_new

    def _update(self, W, dW, mW, vW):
        if isinstance(W[0], list):
            for i in range(len(W)):
                for j in range(len(W[0])):
                    W[i][j], mW[i][j], vW[i][j] = \
                        self._update_element(W[i][j], dW[i][j],
                                           mW[i][j], vW[i][j])
            for i in range(len(W)):
                W[i], mW[i], vW[i] = \
                    self._update_element(W[i], dW[i], mW[i], vW[i])