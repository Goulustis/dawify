from dataclasses import dataclass, field
from typing import Type, Literal
import os
import os.path as osp
import demucs
import torch
import glob

from dawify.config import InstantiateConfig
from dawify.mis_utils import rprint

@dataclass
class DemucModConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: DemucMod)

    out_dir: str = "output"
    """directory to save the results"""

    model_name: Literal["hdemucs_mmi", "htdemucs", "htdemucs_6s", "htdemucs_ft", "mdx", "mdx_extra", "mdx_extra_q", "mdx_q", "repro_mdx_a", "repro_mdx_a_hybrid_only", "repro_mdx_a_time_only"] = "htdemucs"
    """model to use, see https://github.com/facebookresearch/demucs/tree/main/demucs/remote for details"""


class DemucMod:
    def __init__(self, config: DemucModConfig):
        self.config = config
        self.model_name = config.model_name
        self.out_dir = osp.join(config.out_dir, self.model_name)
        self.curr_save_dir = None

        os.makedirs(self.out_dir, exist_ok=True)
        rprint(f"[green]Saved to: {self.out_dir}[/green]")

    @torch.no_grad()
    def seperate(self, inp_f:str):
        file_name = osp.splitext(osp.basename(inp_f))[0]
        self.curr_save_dir = osp.join(self.out_dir, file_name)
        os.makedirs(self.curr_save_dir, exist_ok=True)

        rprint(f"[yellow]Separating {osp.basename(inp_f)}, saving to {self.curr_save_dir}[/yellow]")
        demucs.separate.main([inp_f, "-n", self.model_name, "-j" , "4", self.curr_save_dir])
    
    def get_out_fs(self):
        return sorted(glob.glob(osp.join(self.curr_save_dir, "*.wav")))