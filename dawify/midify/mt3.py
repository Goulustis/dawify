from dataclasses import dataclass, field
from typing import Type, Literal, List
import torch
import os
import os.path as osp

from dawify.dw_config import InstantiateConfig
from dawify.midify.mt3_utils import load_model, process_audio
from dawify.mis_utils import rprint

@dataclass
class MT3Config(InstantiateConfig):
    _target: Type = field(default_factory=lambda: MT3_Mod)

    out_dir: str = "outputs/mt3"

    model_name: Literal["YMT3+", "YPTF+Single (noPS)", "YPTF+Multi (PS)", "YPTF.MoE+Multi (noPS)", "YPTF.MoE+Multi (PS)"] = "YMT3+"
    """model used for transcribing music -> no longer needed"""

    precision: Literal["32", "bf16-mixed", "16"] = "16"
    """precision of the model"""

class MT3_Mod:
    def __init__(self, config: MT3Config):
        self.config = config
        self.out_dir = config.out_dir
        self.model = load_model(config.model_name, config.precision)
        self.curr_save_dir = None

        os.makedirs(self.out_dir, exist_ok=True)
    
    @torch.no_grad()
    def convert(self, inp_f: str):
        """
        inp_f (str): either a mp3 or wav file

        returns:
        str: path to the output midi file
        """
        out_dir = osp.join(self.out_dir, osp.basename(osp.dirname(inp_f)))
        
        file_name = osp.splitext(osp.basename(inp_f))[0]
        pr_name = "/".join(inp_f.split("/")[-2:])
        rprint(f"[yellow]{pr_name} midified to {osp.join(out_dir, file_name)}.mid [/yellow]")

        midi_file = process_audio(self.model, inp_f, out_dir)

        print(midi_file + " is using model: " + str(self.config.model_name) + " with precision: " + str(self.config.precision))
        return midi_file
    
    def conv_list(self, inp_fs: List[str]):
        """
        inp_fs (List[str]): list of either mp3 or wav files
        """
        midi_files = []
        for inp_f in inp_fs:
            if "drums.wav" in inp_f:
                self.config.model_name = "YMT3+"
            elif "guitar.wav" in inp_f:
                self.config.model_name = "YMT3+"
                self.config.precision = "16"
            elif "bass.wav" in inp_f:
                self.config.model_name = "YPTF+Single (noPS)"
            elif "vocals.wav" in inp_f:
                self.config.model_name = "YPTF+Single (noPS)"
            elif "piano.wav" in inp_f:
                self.config.model_name = "YMT3+"
                self.config.precision = "16"
            elif "other.wav" in inp_f:
                self.config.model_name = "YPTF.MoE+Multi (PS)"
            else:
                self.config.model_name = "YMT3+"

            # del self.model
            # torch.cuda.empty_cache()
            self.model = load_model(self.config.model_name, self.config.precision)
            midi_files.append(self.convert(inp_f))
        
        self.curr_save_dir = osp.dirname(midi_files[0])
        return midi_files