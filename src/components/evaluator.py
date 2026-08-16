from collections import defaultdict
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np


class Evaluator:
    """Tracks predictions vs the ground truth the user types in, so we
    can compute accuracy/confusion matrix per model without a separate
    labelled test set."""
    def __init__(self):
        self.y_true = defaultdict(list)
        self.y_pred = defaultdict(list)
        self.model_times = defaultdict(list)

    def add_result(self, true_label, predictions_dict):
        if true_label.strip() == "":
            return

        for model, pred in predictions_dict.items():
            self.y_true[model].append(true_label)
            self.y_pred[model].append(pred)

    def add_timing(self, model_name, time_taken):
        if time_taken is not None:
            self.model_times[model_name].append(time_taken)

    def get_full_results(self):
        results = {}

        for model in self.y_true:
            y_t = self.y_true[model]
            y_p = self.y_pred[model]

            if len(y_t) == 0:
                continue

            acc = accuracy_score(y_t, y_p)
            labels = list(set(y_t + y_p))
            cm = confusion_matrix(y_t, y_p, labels=labels)

            results[model] = {
                "accuracy": acc,
                "confusion_matrix": cm,
                "labels": labels
            }

        avg_times = {
            m: np.mean(t) if len(t) > 0 else 0
            for m, t in self.model_times.items()
        }

        return results, avg_times


