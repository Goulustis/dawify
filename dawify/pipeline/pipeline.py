import tyro
from dataclasses import dataclass, field
from typing import Type

from dawify.dw_config import InstantiateConfig
from dawify.track_split.demuc_mod import DemucModConfig, DemucMod
from dawify.midify.mt3 import MT3Config, MT3_Mod

@tyro.conf.configure(tyro.conf.SuppressFixed)
@dataclass
class PipelineConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Pipeline)

    demuc_config: DemucModConfig = field(default_factory=DemucModConfig)

    mt3_config: MT3Config = field(default_factory=MT3Config)

class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.demuc_mod: DemucMod = config.demuc_config.setup()
        self.midify_mod: MT3_Mod = config.mt3_config.setup()

    def process(self, inp_f:str):
        """
        inp_f (str): path to mp3 file
        """
        self.demuc_mod.seperate(inp_f)

        self.midify_mod.conv_list(self.demuc_mod.get_out_fs())
        