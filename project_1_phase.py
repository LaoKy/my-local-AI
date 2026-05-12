import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(1e-5).sqrt()
        return x / rms * self.weight

class RoPE(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t        = torch.arange(max_len).float()
        freqs    = torch.outer(t, inv_freq)
        emb      = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer('cos', emb.cos())
        self.register_buffer('sin', emb.sin())

    def forward(self, x):
        seq = x.shape[2]
        cos = self.cos[:seq].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:seq].unsqueeze(0).unsqueeze(0)

        x1  = x[..., :x.shape[-1]//2]
        x2  = x[..., x.shape[-1]//2:]
        x_rot = torch.cat([-x2, x1], dim=-1)
        return x * cos + x_rot * sin

class FFN(nn.Module):
    def __init__(self, embed_dim, ffn_dim):
        super().__init__()
        self.W1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.W2 = nn.Linear(ffn_dim, embed_dim, bias=False)
        self.W3 = nn.Linear(embed_dim, ffn_dim, bias=False)

    def forward(self, x):
        gate = F.silu(self.W1(x))
        up   = self.W3(x)
        mid  = gate * up
        return self.W2(mid)

class Attention(nn.Module):
    def __init__(self, embed_dim, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim   = embed_dim // n_heads

        self.wq   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wk   = nn.Linear(embed_dim, self.head_dim * n_kv_heads, bias=False)
        self.wv   = nn.Linear(embed_dim, self.head_dim * n_kv_heads, bias=False)
        self.wo   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.rope = RoPE(self.head_dim)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        scale = self.head_dim ** -0.5
        attn  = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, ffn_dim, n_heads, n_kv_heads, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn  = Attention(embed_dim, n_heads, n_kv_heads)
        self.norm2 = RMSNorm(embed_dim)
        self.ffn   = FFN(embed_dim, ffn_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.norm1(x), mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, ffn_dim,
                 n_layers, n_heads, n_kv_heads,
                 ctx_len, dropout=0.0):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, ffn_dim,n_heads, n_kv_heads, dropout) for _ in range(n_layers)])
        self.norm = RMSNorm(embed_dim)
        self.drop = nn.Dropout(dropout)
        self.ctx_len = ctx_len

    def forward(self, x, mask=None):
        x = self.drop(self.embed(x))

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)

        return x @ self.embed.weight.T

def load_dataset(path, tok, ctx_len):
    samples = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '|' not in line:
                continue
            q, a = line.split('|', 1)
            ids = tok.encode_qa(q.strip(), a.strip())
            if len(ids) < 4:
                continue
            try:
                sep_pos = ids.index(tok.SEP)
            except:
                sep_pos = 0
            if len(ids) > ctx_len + 1:
                ids = ids[:ctx_len + 1]
            samples.append((ids, sep_pos))
    return samples

def make_batch(samples, batch_size, ctx_len, device):
    batch   = random.sample(samples, min(batch_size, len(samples)))
    max_len = min(ctx_len, max(len(s[0]) for s in batch))

    X = torch.zeros(len(batch), max_len, dtype=torch.long)
    Y = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, (ids, sep_pos) in enumerate(batch):
        ids = ids[:max_len+1]
        T   = min(len(ids)-1, max_len)
        X[i, :T] = torch.tensor(ids[:T])
        for t in range(T):
            if t >= sep_pos:
                Y[i, t] = ids[t+1]
    return X.to(device), Y.to(device)

def generate(model, tok, question,
             max_new=40, temperature=0.7, top_k=10):
    model.eval()
    with torch.no_grad():
        ids     = tok.encodeQ(question)
        ids     = torch.tensor([ids], device=DEVICE)
        sep_pos = len(ids[0]) - 1

        for _ in range(max_new):
            T      = ids.shape[1]
            mask   = torch.tril(torch.ones(T, T, device=DEVICE))
            logits = model(ids, mask)[0, -1]  # token cuối

            top_val, top_idx = torch.topk(logits, top_k)
            probs   = F.softmax(top_val / temperature, dim=-1)
            next_id = top_idx[torch.multinomial(probs, 1)]

            ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)

            if next_id.item() == tok.EOS:
                break

        answer_ids = ids[0, sep_pos+1:].tolist()
        return tok.decode_answer(answer_ids)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CFG = {
    'vocab_size': 2000,
    'embed_dim':  256,
    'ffn_dim':    1024,
    'n_layers':   4,
    'n_heads':    4,
    'n_kv_heads': 2,
    'ctx_len':    96,
    'dropout':    0.1,
    'lr':         1e-3,
    'epochs':     500,
    'batch_size': 16,
}

exec(open('tokenizer.py', encoding='utf-8').read())
tok = BPETokenizer(vocab_size=CFG['vocab_size'])
tok.load('data/tokenizer.json')
CFG['vocab_size'] = len(tok.vocab)

samples = load_dataset('data/dataset.txt', tok, CFG['ctx_len'])

model = MiniGPT(
    vocab_size = CFG['vocab_size'],
    embed_dim  = CFG['embed_dim'],
    ffn_dim    = CFG['ffn_dim'],
    n_layers   = CFG['n_layers'],
    n_heads    = CFG['n_heads'],
    n_kv_heads = CFG['n_kv_heads'],
    ctx_len    = CFG['ctx_len'],
    dropout    = CFG['dropout'],
).to(DEVICE)

print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CFG['lr'],
    weight_decay=0.01
)

model.train()
for epoch in range(1, CFG['epochs']+1):
    X, Y = make_batch(samples, CFG['batch_size'],
                      CFG['ctx_len'], DEVICE)
    T    = X.shape[1]
    mask = torch.tril(torch.ones(T, T, device=DEVICE))

    logits = model(X, mask)
    loss   = F.cross_entropy(
        logits.view(-1, CFG['vocab_size']),
        Y.view(-1),
        ignore_index=-100
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")

print(generate(model, tok, "xin chào"))
print(generate(model, tok, "bạn là ai"))