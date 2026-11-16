import random
import warnings

from .deduplication import duplicate_groups
from .validation import LABELS, validate


def _allocation(count):
    if count < 3:
        return count, 0, 0
    validation = max(1, round(count * .15))
    test = max(1, round(count * .15))
    return count - validation - test, validation, test


def split_dataset(df, seed=42, threshold=.9, languages=None):
    """Split duplicate/related groups without allowing a group across partitions."""
    validate(df, True, languages)
    working=df.copy(); working["duplicate_group"]=duplicate_groups(working, threshold)
    group_labels=working.groupby("duplicate_group").sentiment.agg(lambda values: sorted(set(values)))
    conflicts=group_labels[group_labels.map(len).ne(1)]
    if not conflicts.empty:
        raise ValueError(f"Related or duplicate examples have conflicting sentiment labels in groups: {conflicts.index.tolist()}")
    if len(group_labels)<7: raise ValueError("Need at least 7 independent groups for three splits")
    rng=random.Random(seed)
    assigned={"train":[],"validation":[],"test":[]}
    for label in LABELS:
        groups=group_labels[group_labels.map(lambda values: values[0]==label)].index.tolist()
        rng.shuffle(groups)
        n_train,n_validation,n_test=_allocation(len(groups))
        if not n_validation or not n_test:
            warnings.warn(f"Sentiment '{label}' has only {len(groups)} independent groups; it cannot appear in every split. Inspect class support.")
        assigned["train"].extend(groups[:n_train])
        assigned["validation"].extend(groups[n_train:n_train+n_validation])
        assigned["test"].extend(groups[n_train+n_validation:n_train+n_validation+n_test])
    outputs={name:working[working.duplicate_group.isin(groups)].assign(split=name).copy() for name,groups in assigned.items()}
    if sum(map(len,outputs.values()))!=len(working): raise RuntimeError("Split allocation lost one or more rows")
    return outputs
