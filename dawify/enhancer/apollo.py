import os.path as osp
import glob
import os
from dataclasses import dataclass, field
from typing import Type, List
import shutil

from dawify.dw_config import InstantiateConfig
from dawify.enhancer.apollo_utils import load_model, inference, MODEL_PATH
from dawify.mis_utils import rprint

avail_models = ", ".join(list(MODEL_PATH.keys()))
@dataclass
class ApolloConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Apollo)

    model_name:str = "pytorch"
    f"""available models: {avail_models}"""

    chunk_size:int = 10
    """no idea what this is"""

    overlap:int = 2
    """no idea what this is"""

    out_dir:str = "outputs/apollo"
    """directory to save the results"""


class Apollo:
    def __init__(self, config: ApolloConfig):
        self.config = config
        self.out_dir = config.out_dir
        self.model = load_model(config.model_name)
        self.model.eval()
        self.curr_save_dir = None

        os.makedirs(self.out_dir, exist_ok=True)

    def enhance(self, input_wav):

        self.curr_save_dir = osp.join(self.out_dir, osp.basename(osp.dirname(input_wav)))
        os.makedirs(self.curr_save_dir, exist_ok=True)

        file_name = osp.basename(input_wav)
        out_wav = osp.join(self.curr_save_dir, file_name)

        rprint(f"[yellow]enhancing {file_name} to {out_wav}[/yellow]")
        inference(input_wav, out_wav, self.model, chunk_size=self.config.chunk_size, overlap=self.config.overlap)
    
    def enhance_list(self, input_wavs: List[str]):
        for input_wav in input_wavs:

            # NOTE: ignore bass files
            if "bass" in input_wav:
                shutil.copy(input_wav, self.curr_save_dir)

            self.enhance(input_wav)
    
    def get_out_fs(self):
        return sorted(glob.glob(osp.join(self.curr_save_dir, "*.wav")))