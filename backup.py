import sys
from collections import Counter
from typing import List, Tuple, Union

data: List[List[str]] = []
all_lists: List[List[List[str]]] = []
feature_list: List[Tuple[int, str]] = []


def load_dataset(filename: str) -> List[List[List[str]]]:
    print(f"Loading dataset from file {filename}\n")
    with open(filename) as infile:
        for line in infile:
            line = line.strip()
            line = line.split(',')
            data.append(line)
    return k_fold_data(data)


def k_fold_data(data: List[List[str]]) -> List[List[List[str]]]:
    for fold in range(1, 11):
        next_fold = fold
        fold_list: List[List[str]] = []
        for i, row in enumerate(data):
            if i == next_fold:
                next_fold += 10
                fold_list.append(row)
        all_lists.append(fold_list)
    return all_lists


def mu() -> List[Tuple[int, float]]:
    mean_features = []
    feature_sum = Counter()
    feature_count = Counter()
    total = 4601

    for key, value in feature_list:
        feature_sum[key] += float(value)
        feature_count[key] += 1

    for key in sorted(feature_sum.keys()):
        mean_features.append((key, feature_sum[key] / total))

    return mean_features


def sd(mean_features: List[Tuple[int, float]]) -> List:
    sd_features = []
    sum_squares = Counter()

    for id, mean_value in mean_features:
        for key, value in feature_list:
            if id == key:
                difference = float(value) - mean_value
                difference_sqr = difference * difference
                sum_squares[key] += difference_sqr

    return [sum_squares[key] for key in sorted(sum_squares.keys())]


def pre_condition(folded_data: List[List[List[str]]]) -> None:
    count = 0

    for i in range(10):
        for fold_list in folded_data[i]:
            count += 1
            for b, feature_value in enumerate(fold_list):
                feature_data = b, feature_value
                feature_list.append(feature_data)

    mean_features = mu()
    print(mean_features)
    feature_sd_list = sd(mean_features)
    print(feature_sd_list)
    print(count)


if __name__ == '__main__':
    folded_data = load_dataset("data/spambase.data")
    z_score_data = pre_condition(folded_data)
