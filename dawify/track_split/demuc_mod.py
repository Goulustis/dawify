from dataclasses import dataclass, field
from typing import Type, Literal
import os
import os.path as osp
import demucs.separate
import torch
import glob
import shlex

from dawify.dw_config import InstantiateConfig
from dawify.mis_utils import rprint

@dataclass
class DemucModConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: DemucMod)

    ignore_exist: bool = True
    """if true, will not process file that are altready processed"""

    out_dir: str = "outputs/demuc"
    """directory to save the results"""

    model_name: Literal["hdemucs_mmi", "htdemucs", "htdemucs_6s", "htdemucs_ft", "mdx", "mdx_extra", "mdx_extra_q", "mdx_q", "repro_mdx_a", "repro_mdx_a_hybrid_only", "repro_mdx_a_time_only"] = "htdemucs"
    """model to use, see https://github.com/facebookresearch/demucs/tree/main/demucs/remote for details"""


class DemucMod:
    def __init__(self, config: DemucModConfig):
        self.config = config
        self.model_name = config.model_name
        self.out_dir = config.out_dir
        self.ignore_exist = config.ignore_exist
        self.curr_save_dir = None

        os.makedirs(self.out_dir, exist_ok=True)
        rprint(f"[green]Saving demucs to: {self.out_dir}[/green]")

    @torch.no_grad()
    def seperate(self, inp_f:str):
        file_name = osp.splitext(osp.basename(inp_f))[0]
        self.curr_save_dir = osp.join(self.out_dir, self.model_name, file_name)

        if self.ignore_exist and osp.exists(self.curr_save_dir):
            rprint(f"[yellow]Skipping {osp.basename(inp_f)}, already processed[/yellow]")
            return

        os.makedirs(self.curr_save_dir, exist_ok=True)

        rprint(f"[yellow]Separating {osp.basename(inp_f)}, saving to {self.curr_save_dir}[/yellow]")
        # NOTE: demucs will save to self.out_dir/model_name/track_name
        demucs.separate.main(shlex.split(f'"{inp_f}" -n htdemucs -j 4 --out "{self.out_dir}"'))
    
    def get_out_fs(self):
        return sorted(glob.glob(osp.join(self.curr_save_dir, "*.wav")))