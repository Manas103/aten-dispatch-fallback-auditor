import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from task import N_CLASSES, make_dataset


def test_dataset_is_seeded_and_deterministic():
    ids_a, labels_a = make_dataset(50, seed=42)
    ids_b, labels_b = make_dataset(50, seed=42)
    assert (ids_a == ids_b).all()
    assert (labels_a == labels_b).all()


def test_different_seeds_give_different_data():
    ids_a, _ = make_dataset(50, seed=42)
    ids_b, _ = make_dataset(50, seed=43)
    assert not (ids_a == ids_b).all()


def test_label_is_first_token_mod_n_classes():
    ids, labels = make_dataset(100, seed=1)
    assert (labels == ids[:, 0] % N_CLASSES).all()
    assert labels.min() >= 0
    assert labels.max() < N_CLASSES
