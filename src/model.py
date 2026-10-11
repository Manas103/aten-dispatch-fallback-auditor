"""A small transformer encoder block, used only as a realistic forward
and backward step for the dispatch auditor to trace. Deliberately mixes
operators with a direct CPU kernel (matmul, addmm, relu, native_layer_norm,
softmax's own _softmax kernel) with operators that resolve through a
Composite (decomposing) kernel on this PyTorch build (linear, layer_norm,
gelu's erf path in some builds, cross_entropy) -- see docs/op_classification.md
for which is which and why, measured rather than assumed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 1000
D_MODEL = 128
N_HEADS = 1
HEAD_DIM = D_MODEL // N_HEADS
MLP_HIDDEN = 4 * D_MODEL
SEQ_LEN = 16
BATCH = 4


class EncoderBlock(nn.Module):
    """One combined QKV linear (GPT-2-style) instead of three separate
    projections, and the output projection folded into the MLP's first
    linear's input width: fewer nn.Linear calls per block means fewer
    aten::t / addmm_backward calls per step, which otherwise dominate the
    op count (one aten.t call alone showed up 35 times for a 7-linear,
    1-layer block during this project's design pass -- see
    docs/op_classification.md)."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.wqkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.fc1 = nn.Linear(D_MODEL, MLP_HIDDEN)
        self.fc2 = nn.Linear(MLP_HIDDEN, D_MODEL)

    def attn(self, x):
        qkv = self.wqkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        scores = (q @ k.transpose(-2, -1)) / (HEAD_DIM ** 0.5)
        probs = F.softmax(scores, dim=-1)
        return probs @ v

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class TinyEncoder(nn.Module):
    """Token + positional embedding (aten::embedding, see
    docs/op_classification.md -- a real fallback this model actually
    exercises, unlike aten::linear/aten::layer_norm which this build's
    Python-level F.linear/F.layer_norm bypass entirely before the
    dispatcher ever sees the generic composite op) feeding into the
    encoder blocks."""

    def __init__(self, n_layers=1, n_classes=10):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos_emb = nn.Embedding(SEQ_LEN, D_MODEL)
        self.blocks = nn.ModuleList([EncoderBlock() for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, n_classes)

    def forward(self, token_ids):
        B, T = token_ids.shape
        pos = torch.arange(T).unsqueeze(0)
        x = self.tok_emb(token_ids) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x[:, 0, :])


def build_step_inputs(seed=1234):
    torch.manual_seed(seed)
    model = TinyEncoder()
    token_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    labels = torch.randint(0, 10, (BATCH,))
    return model, token_ids, labels


def run_forward_backward_step(model, token_ids, labels):
    logits = model(token_ids)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    return loss
