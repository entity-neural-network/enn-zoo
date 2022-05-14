from os import listdir
from pathlib import Path

import hyperstate
from hyperstate.schema.schema_change import Severity
from hyperstate.schema.schema_checker import SchemaChecker
from hyperstate.schema.types import load_schema

from enn_trainer.config import TrainConfig



def test_configs() -> None:
    for config in ["entity-gym", "procgen", "codecraft"]:
        config_dir = Path(__file__).parent.parent / "configs" / config
        for config_file in listdir(config_dir):
            print(config_file)
            hyperstate.load(TrainConfig, config_dir / config_file)
