from dataclasses import dataclass, field
from typing import Type, Literal, List
import torch
import os

from dawify.dw_config import InstantiateConfig
from dawify.midify.mt3_utils import load_model, process_audio

@dataclass
class MT3Config(InstantiateConfig):
    _target: Type = field(default_factory=lambda: MT3_Mod)

    out_dir: str = "outputs/mt3"

    model_name: Literal["YMT3+", "YPTF+Single (noPS)", "YPTF+Multi (PS)", "YPTF.MoE+Multi (noPS)", "YPTF.MoE+Multi (PS)"] = "YMT3+"
    """model used for transcribing music -> midi"""

    precision: Literal["32", "bf16-mixed", "16"] = "16"
    """precision of the model"""

class MT3_Mod:
    def __init__(self, config: MT3Config):
        self.config = config
        self.out_dir = config.out_dir
        self.model = load_model(config.model_name, config.precision)

        os.makedirs(self.out_dir, exist_ok=True)
    
    @torch.no_grad()
    def convert(self, inp_f: str):
        """
        inp_f (str): either a mp3 or wav file
        """
        midi_file = process_audio(self.model, inp_f)

        return midi_file
    
    def conv_list(self, inp_fs: List[str]):
        """
        inp_fs (List[str]): list of either mp3 or wav files
        """
        midi_files = []
        for inp_f in inp_fs:
            midi_files.append(self.convert(inp_f))
        
        return midi_files