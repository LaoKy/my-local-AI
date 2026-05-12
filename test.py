import torch
import torch.nn as nn
import torch.nn.functional as F
import random

CFG = {
    'vocab_size':  2000,
    'embed_dim':   352,
    'ffn_dim':     1408,
    'n_layers':    6,
    'n_heads':     4,
    'n_kv_heads':  2,
    'ctx_len':     128,
    'dropout':     0.1,
}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
params = (CFG['vocab_size'] * CFG['embed_dim']
        + CFG['n_layers'] * (CFG['embed_dim']**2 * 4
                           + CFG['embed_dim'] * CFG['ffn_dim'] * 3))

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✅ GPU: {gpu_name} | VRAM: {vram:.1f} GB")
else:
    print("⚠️  CPU — nên bật GPU trong Runtime settings")

print(f"📐 Config: {CFG['n_layers']} layers | {CFG['n_heads']} heads | {CFG['embed_dim']} dim")
print(f"🧮 Ước tính params: ~{params/1e6:.1f}M")


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.gamma

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
        return x * cos + torch.cat([-x2, x1], dim=-1) * sin

class GQAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim//n_heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, self.head_dim*n_kv_heads, bias=False)
        self.wv = nn.Linear(dim, self.head_dim*n_kv_heads, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.rope = RoPE(self.head_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = self.rope(q)
        k = self.rope(k)
        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        attn = (q @ k.transpose(-2, -1)) * self.head_dim**(-0.5)
        if mask is not None:
            attn = attn.masked_fill(mask==0, float('-inf'))
        attn = self.attn_drop(torch.softmax(attn, dim=-1))
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)

class SwiGLU(nn.Module):
    def __init__(self, dim, ffn_dim, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg['embed_dim'])
        self.attn = GQAttention(cfg['embed_dim'], cfg['n_heads'], cfg['n_kv_heads'], cfg['dropout'])
        self.norm2 = RMSNorm(cfg['embed_dim'])
        self.ffn = SwiGLU(cfg['embed_dim'], cfg['ffn_dim'], cfg['dropout'])
        self.drop = nn.Dropout(cfg['dropout'])
    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.norm1(x), mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x

class MiniGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(cfg['vocab_size'], cfg['embed_dim'])
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
        self.norm = RMSNorm(cfg['embed_dim'])
        self.drop = nn.Dropout(cfg['dropout'])
        self.cfg = cfg
    def forward(self, x, mask=None):
        x = self.drop(self.embed(x))
        for blocks in self.blocks:
            x = blocks(x, mask)
        x = self.norm(x)
        return x @ self.embed.weight.T


def make_batch(samples, batch_size, ctx_len, device):
    batch = random.sample(samples, min(batch_size, len(samples)))
    max_len = min(ctx_len, max(len(s) for s in batch))
    X = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
    Y = torch.full((len(batch), max_len), -100, dtype=torch.long, device=device)
    for i, ids in enumerate(batch):
        T = min(len(ids) - 1, max_len)
        X[i, :T] = torch.tensor(ids[:T], dtype=torch.long)
        Y[i, :T] = torch.tensor(ids[1:T+1], dtype=torch.long)
    return X, Y

samples = [
    list(range(i, i + 30))
    for i in range(200)
]

model = MiniGPT(CFG).to(device)
best_loss = float('inf')
MAX_EPOCHS = 300
batch_size = 16
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
model.train()
for epoch in range(1, MAX_EPOCHS+1):
    X, Y = make_batch(samples, batch_size, CFG['ctx_len'], device)
    T = X.shape[1]
    mask = torch.tril(torch.ones(T, T, device=device))
    mask = mask.unsqueeze(0).unsqueeze(0)
    logits = model(X, mask)
    loss = F.cross_entropy(logits.view(-1, CFG['vocab_size']), Y.view(-1), ignore_index=-100)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    if epoch % 50 == 0:
        current_lr = scheduler.get_last_lr()[0]
        print(f"epoch: {epoch}, loss: {loss.item():.4f}, lr: {current_lr:.6f}")
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save({'model_state': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'loss': best_loss}, 'ckpt.pt')
        print(f"Checkpoint saved at epoch {epoch} with loss {best_loss:.4f}")


