from collections import defaultdict
from libs.ts import augment
import pickle
import random

def group_by_class(dataset):
    """
    Group samples in the dataset by their class label.

    Returns:
        dict: Dictionary mapping class labels to lists of samples.
    """

    groups = defaultdict(list)
    for data in dataset:
        label = data[1]
        sample = data
        groups[label].append(sample)
    return dict(groups)

def balance_classes(groups, samples_per_class):
    """
    Balances each class to have the same number of samples by either trimming or augmenting.

    Args:
        groups (dict): Dictionary mapping class labels to lists of samples.

    Returns:
        dict: Balanced dictionary with each class having `samples_per_class` samples.
    """

    balanced = {}
    for label, samples in groups.items():
        if len(samples) == samples_per_class:
            balanced[label] = samples
        else:
            balanced[label] = samples.copy()
            choice = random.randint(0, len(samples) - 1)
            prec, curr, succ = samples[choice][2], samples[choice][3], samples[choice][4]
            for _ in range(min(10, samples_per_class - len(samples))):
                prec_aug, curr_aug, succ_aug = augment(prec, curr, succ)
                balanced[label].append((samples[choice][0], samples[choice][1], prec_aug, curr_aug, succ_aug))
    return balanced


with open(f"dataset/train.pkl", 'rb') as f:
    train_data = pickle.load(f)

groups = group_by_class(train_data)

balanced = balance_classes(groups, 158)

train_data = []

for label, samples in balanced.items():
    train_data.extend(samples)

with open(f"dataset/augmented.pkl", 'wb') as f:
    pickle.dump(train_data, f)
