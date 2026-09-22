"""Observer pattern: training events are broadcast to subscribers."""
from collections import defaultdict


class TrainingMonitor:
    def __init__(self):
        self._subs = []

    def attach(self, observer):
        self._subs.append(observer)

    def notify(self, event, **payload):
        for s in self._subs:
            s.update(event, **payload)


class ConsoleLogger:
    def __init__(self, every=10):
        self.every = every

    def update(self, event, **p):
        if event == "epoch_end" and (p["epoch"] + 1) % self.every == 0:
            print(f"  epoch {p['epoch'] + 1:3d}  loss {p['loss']:.4f}")


class HistoryRecorder:
    def __init__(self):
        self.history = defaultdict(list)

    def update(self, event, **p):
        if event == "epoch_end":
            self.history["loss"].append(p["loss"])
