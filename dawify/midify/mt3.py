from dataclasses import dataclass, field
from typing import Type, Literal
import torch

from dawify.config import InstantiateConfig
from dawify.midify.mt3_utils import load_model, process_audio

@dataclass
class MT3Config(InstantiateConfig):
    _target: Type = field(default_factory=lambda: MT3_Mod)

    model_name: Literal["YMT3+", "YPTF+Single (noPS)", "YPTF+Multi (PS)", "YPTF.MoE+Multi (noPS)", "YPTF.MoE+Multi (PS)"] = "YMT3+"
    """model used for transcribing music -> midi"""

    precision: Literal["32", "bf16-mixed", "16"] = "16"
    """precision of the model"""

class MT3_Mod:
    def __init__(self, config: MT3Config):
        self.config = config
        self.model = load_model(config.model_name, config.precision)
    
    @torch.no_grad()
    def convert(self, inp_f: str):
        """
        inp_f (str): either a mp3 or wav file
        """
        midi_file = process_audio(self.model, inp_f)

        return midi_file