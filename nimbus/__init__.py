from nimbus.utils.types import STAGE_PIPE

__all__ = ["DataEngine", "DistPipeDataEngine", "run_data_engine"]


def __getattr__(name):
    if name in {"DataEngine", "DistPipeDataEngine"}:
        from .data_engine import DataEngine, DistPipeDataEngine

        globals()["DataEngine"] = DataEngine
        globals()["DistPipeDataEngine"] = DistPipeDataEngine
        return globals()[name]
    raise AttributeError(f"module 'nimbus' has no attribute {name!r}")


def run_data_engine(config, master_seed=None):
    import ray
    import nimbus_extension  # noqa: F401  pylint: disable=unused-import

    if STAGE_PIPE in config:
        from .data_engine import DistPipeDataEngine

        ray.init(num_gpus=1)
        data_engine = DistPipeDataEngine(config, master_seed=master_seed)
    else:
        from .data_engine import DataEngine

        data_engine = DataEngine(config, master_seed=master_seed)
    data_engine.run()
