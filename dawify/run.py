import tyro
import logging

# Configure logging at the start of your program
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from dawify.pipeline.pipeline import PipelineConfig, Pipeline
from dawify.midify.mt3 import MT3Config

def main_cli(config:PipelineConfig, inp_f:str):
    config = PipelineConfig(
        mt3_config=MT3Config(
            model_name="YPTF.MoE+Multi (noPS)",
            precision="32"
        )
    )

    pipeline:Pipeline = config.setup()
    pipeline.process(inp_f)


def main_script():
    config = PipelineConfig(
        mt3_config=MT3Config(
            model_name="YPTF.MoE+Multi (noPS)",
            precision="32"
        )
    )

    pipeline:Pipeline = config.setup()
    pipeline.process("/home/kaiyolau2/Code/dawify/assets/sample_level1.wav")


if __name__ == "__main__":
    # main_script()       # Run this script with `python dawify/run.py`
    tyro.cli(main_cli)    # Run this script with `python dawify/run.py --help`
