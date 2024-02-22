import os
from dataclasses import dataclass
from typing import List


@dataclass
class DataStats:
    """
    Data statistics
    """
    count_labeled: int
    count_unlabeled: int


@dataclass
class TrainIterationStats:
    iteration: int
    loss: List[float]
    speed: List[float]
    accuracy: float
    data_stats_at_beginning: DataStats
    data_stats_at_end: DataStats


@dataclass
class TrainStats:
    number_of_iterations: int
    stats: List[TrainIterationStats]


IS_TRAIN_LOG = lambda filename: filename.startswith("train_2")
IS_TRAIN_ITERATION_LOG = lambda filename: filename.startswith("train_iteration")


def build_train_stats(directory: str) -> TrainStats:
    files = os.listdir(directory)
    train_file = list(filter(IS_TRAIN_LOG, files))[0]
    train_iteration_files = sorted(list(filter(IS_TRAIN_ITERATION_LOG, files)))
    with open(os.path.join(directory, train_file), "r") as train_log_file:
        lines = train_log_file.readlines()
        iteration_indexes = [i for i, line in enumerate(lines) if "Training iteration" in line]
        iterations = []
        for i, iteration_index in enumerate(iteration_indexes):
            start_log_items = lines[iteration_index].split(" ")
            iteration_key = int(start_log_items[5])
            count_labeled_at_start = int(start_log_items[7])
            evaluation_log = lines[iteration_index + 1].split(" ")
            accuracy = float(evaluation_log[9].split("}")[0])
            generated_log = lines[iteration_index + 2].split(" ")
            count_pseudo_labels = int(generated_log[4])
            count_unlabeled_at_start = int(generated_log[8])
            data_stats_at_beginning = DataStats(count_labeled_at_start, count_unlabeled_at_start)
            data_stats_at_end = DataStats(count_labeled_at_start + count_pseudo_labels, count_unlabeled_at_start
                                          - count_pseudo_labels)
            iteration_stats = build_iteration_stats(os.path.join(directory, train_iteration_files[i]), iteration_key,
                                                    accuracy, data_stats_at_beginning, data_stats_at_end)
            iterations.append(iteration_stats)
        return TrainStats(number_of_iterations=len(iteration_indexes), stats=iterations)


def build_iteration_stats(filename: str, iteration_key: int, accuracy: float, data_beginning: DataStats,
                          data_end: DataStats) -> TrainIterationStats:
    with open(filename, "r") as train_iteration_file:
        lines = train_iteration_file.readlines()
        epoch_indexes = [i for i, line in enumerate(lines) if "diffusion epoch" in line]
        speeds = []
        losses = []
        for epoch_index in epoch_indexes:
            epoch_tokens = lines[epoch_index].split()
            speed = float(epoch_tokens[10].replace("it/s,", ""))
            loss = float(epoch_tokens[-1].split("=")[1].split("]")[0])
            speeds.append(speed)
            losses.append(loss)
        return TrainIterationStats(iteration=iteration_key, speed=speeds, loss=losses, accuracy=accuracy,
                                   data_stats_at_beginning=data_beginning, data_stats_at_end=data_end)


if __name__ == "__main__":
    result = build_train_stats("logs_test/")
    print()
