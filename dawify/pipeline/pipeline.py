import tyro
from dataclasses import dataclass, field
from typing import Type

from dawify.config import InstantiateConfig
from dawify.track_split.demuc_mod import DemucModConfig
from dawify.midify.mt3 import MT3Config

@tyro.conf.configure(tyro.conf.SuppressFixed)
@dataclass
class PipelineConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Pipeline)

    demuc_config: DemucModConfig = field(default_factory=DemucModConfig)

    mt3_config: MT3Config = field(default_factory=MT3Config)

class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.demuc_mod = config.demuc_config.setup()
        self.midify_mod = config.mt3_config.setup()

    def run(self, inp_f:str):
        """
        inp_f (str): path to mp3 file
        """
        self.demuc_mod.seperate(inp_f)

    def __call__(self):
        self.run()