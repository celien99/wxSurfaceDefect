__all__ = ["HRTrainer"]


def __getattr__(name):
    if name == "HRTrainer":
        from .trainer import HRTrainer

        return HRTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
