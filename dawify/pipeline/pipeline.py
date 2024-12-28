
# pipeline.py
import tyro
from dataclasses import dataclass, field
from typing import Type

from dawify.dw_config import InstantiateConfig
from dawify.track_split.demuc_mod import DemucModConfig, DemucMod
from dawify.midify.mt3 import MT3Config, MT3_Mod
from dawify.pre_processing.audioPreProcessor import AudioPreProcessor, AudioPreProcessorConfig
from dawify.enhancer.apollo import ApolloConfig, Apollo

@tyro.conf.configure(tyro.conf.SuppressFixed)
@dataclass
class PipelineConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Pipeline)

    demuc_config: DemucModConfig = field(default_factory=DemucModConfig)

    enhancer_config: ApolloConfig = field(default_factory=ApolloConfig)

    mt3_config: MT3Config = field(default_factory=MT3Config)
    
    audio_preproc_config: AudioPreProcessorConfig = field(default_factory=AudioPreProcessorConfig)

class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.demuc_mod: DemucMod = config.demuc_config.setup()
        self.midify_mod: MT3_Mod = config.mt3_config.setup()
        self.audio_preprocessor: AudioPreProcessor = config.audio_preproc_config.setup()
        self.enhancer: Apollo = config.enhancer_config.setup()

    def process(self, inp_f: str):
        """
        inp_f (str): path to mp3 file
        """
        # Step 1: Demuc separation
        self.demuc_mod.seperate(inp_f)
        separated_files = self.demuc_mod.get_out_fs()

        # setp 1.5: Enhance separated files
        # self.enhancer.enhance_list(separated_files)
        # separated_files = self.enhancer.get_out_fs()

        # Step 2: Process all splited audio through AudioPreProcessor
        preprocessed_files = self.audio_preprocessor.process(separated_files)


        # Step 3: Midify conversion
        self.midify_mod.conv_list(preprocessed_files)

    def get_in_out_dirs(self):
        return self.demuc_mod.curr_save_dir, self.midify_mod.curr_save_dir