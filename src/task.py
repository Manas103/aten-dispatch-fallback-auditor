"""A synthetic, disclosed-as-synthetic classification task for the
OpenVINO/NNCF leg: `TinyEncoder` (model.py) is otherwise randomly
initialized and never trained, so there is no real top-1 accuracy to
budget an INT8 regression against. The label is
`token_ids[:, 0] % n_classes`: classify the sequence by its first token's
value mod 10. A first attempt used `sum(token_ids) % n_classes`
(requiring the encoder to aggregate an exact value across all 16
positions); train loss fell but val accuracy stayed at chance (about
0.09), because a 1-layer attention block has no exact-arithmetic
mechanism and the model was memorizing training sums instead of learning
the mod-10 rule. The first-token task still exercises the full encoder
(the classification head reads position 0 only after the attention block
and residual connections have mixed in every other position's
embedding, so position 0's representation is not just its own raw
embedding), but is actually learnable by this model size.
"""
import torch

from model import BATCH, SEQ_LEN, VOCAB

N_CLASSES = 10


def make_dataset(n_examples: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    token_ids = torch.randint(0, VOCAB, (n_examples, SEQ_LEN), generator=g)
    labels = token_ids[:, 0] % N_CLASSES
    return token_ids, labels


def iterate_batches(token_ids: torch.Tensor, labels: torch.Tensor, batch_size: int = BATCH):
    n = token_ids.shape[0]
    for start in range(0, n - batch_size + 1, batch_size):
        end = start + batch_size
        yield token_ids[start:end], labels[start:end]
