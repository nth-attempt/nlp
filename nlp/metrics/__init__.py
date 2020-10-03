from collections import defaultdict


class Metric(object):
    # https://github.com/flairNLP/flair/blob/master/flair/training_utils.py
    def __init__(self, name, beta=1):
        self.name = name
        self.beta = beta
        self.initialize()

    def initialize(self):
        self._tps = defaultdict(int)
        self._fps = defaultdict(int)
        self._tns = defaultdict(int)
        self._fns = defaultdict(int)
        self._supports = defaultdict(int)
        self._classes = set()

    def add_tp(self, class_name):
        self._tps[class_name] += 1

    def add_tn(self, class_name):
        self._tns[class_name] += 1

    def add_fp(self, class_name):
        self._fps[class_name] += 1

    def add_fn(self, class_name):
        self._fns[class_name] += 1

    def get_tp(self, class_name=None):
        if class_name is None:
            return sum(
                [self._tps[class_name] for class_name in self.get_classes()]
            )
        return self._tps[class_name]

    def get_tn(self, class_name=None):
        if class_name is None:
            return sum(
                [self._tns[class_name] for class_name in self.get_classes()]
            )
        return self._tns[class_name]

    def get_fp(self, class_name=None):
        if class_name is None:
            return sum(
                [self._fps[class_name] for class_name in self.get_classes()]
            )
        return self._fps[class_name]

    def get_fn(self, class_name=None):
        if class_name is None:
            return sum(
                [self._fns[class_name] for class_name in self.get_classes()]
            )
        return self._fns[class_name]

    def precision(self, class_name=None):
        if self.get_tp(class_name) + self.get_fp(class_name) > 0:
            return self.get_tp(class_name) / (
                self.get_tp(class_name) + self.get_fp(class_name)
            )
        return 0.0

    def recall(self, class_name=None):
        if self.get_tp(class_name) + self.get_fn(class_name) > 0:
            return self.get_tp(class_name) / (
                self.get_tp(class_name) + self.get_fn(class_name)
            )
        return 0.0

    def f_score(self, class_name=None):
        if self.precision(class_name) + self.recall(class_name) > 0:
            return (
                (1 + self.beta * self.beta)
                * (self.precision(class_name) * self.recall(class_name))
                / (
                    self.precision(class_name) * self.beta * self.beta
                    + self.recall(class_name)
                )
            )
        return 0.0

    def accuracy(self, class_name=None):
        if (
            self.get_tp(class_name)
            + self.get_fp(class_name)
            + self.get_fn(class_name)
            + self.get_tn(class_name)
            > 0
        ):
            return (self.get_tp(class_name) + self.get_tn(class_name)) / (
                self.get_tp(class_name)
                + self.get_fp(class_name)
                + self.get_fn(class_name)
                + self.get_tn(class_name)
            )
        return 0.0

    def micro_avg_f_score(self):
        return self.f_score(None)

    def macro_avg_f_score(self):
        class_f_scores = [
            self.f_score(class_name) for class_name in self.get_classes()
        ]
        if len(class_f_scores) == 0:
            return 0.0
        macro_f_score = sum(class_f_scores) / len(class_f_scores)
        return macro_f_score

    def micro_avg_accuracy(self):
        return self.accuracy(None)

    def macro_avg_accuracy(self):
        class_accuracy = [
            self.accuracy(class_name) for class_name in self.get_classes()
        ]

        if len(class_accuracy) > 0:
            return sum(class_accuracy) / len(class_accuracy)

        return 0.0

    def get_classes(self):
        classes = sorted(list(self._classes))
        return classes

    def to_tsv(self):
        return "{}\t{}\t{}\t{}".format(
            self.precision(),
            self.recall(),
            self.accuracy(),
            self.micro_avg_f_score(),
        )

    @staticmethod
    def tsv_header(prefix=None):
        if prefix:
            return "{0}_PRECISION\t{0}_RECALL\t{0}_ACCURACY\t{0}_F-SCORE".format(
                prefix
            )

        return "PRECISION\tRECALL\tACCURACY\tF-SCORE"

    @staticmethod
    def to_empty_tsv():
        return "\t_\t_\t_\t_"

    def __str__(self):
        all_classes = self.get_classes()
        all_classes = [None] + all_classes
        all_lines = [
            "{0:<10}\ttp: {1} - fp: {2} - fn: {3} - tn: {4} - precision: {5:.4f} - recall: {6:.4f} - accuracy: {7:.4f} - f1-score: {8:.4f}".format(
                self.name if class_name is None else class_name,
                self.get_tp(class_name),
                self.get_fp(class_name),
                self.get_fn(class_name),
                self.get_tn(class_name),
                self.precision(class_name),
                self.recall(class_name),
                self.accuracy(class_name),
                self.f_score(class_name),
            )
            for class_name in all_classes
        ]
        return "\n".join(all_lines)

    def score(self, y_true, y_pred, ignore_classes=[]):
        self.initialize()

        for gold_tags, predicted_tags in zip(y_true, y_pred):
            for tag, prediction in predicted_tags:
                if tag not in ignore_classes:
                    if tag not in self._classes:
                        self._classes.add(tag)

                    if (tag, prediction) in gold_tags:
                        self.add_tp(tag)
                    else:
                        self.add_fp(tag)

            for tag, gold in gold_tags:
                if tag not in ignore_classes:
                    if tag not in self._classes:
                        self._classes.add(tag)

                    if (tag, gold) not in predicted_tags:
                        self.add_fn(tag)
                    else:
                        self.add_tn(tag)
