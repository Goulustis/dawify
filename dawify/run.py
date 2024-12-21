import tyro

from dawify.pipeline.pipeline import PipelineConfig, Pipeline
from dawify.midify.mt3 import MT3Config

def main_cli(config:PipelineConfig, inp_f:str):
    config = PipelineConfig(
        mt3_config=MT3Config(
            model_name="YMT3+",
            precision="16"
        )
    )

    pipeline:Pipeline = config.setup()
    pipeline.process(inp_f)


def main_script():
    config = PipelineConfig(
        mt3_config=MT3Config(
            model_name="YMT3+",
            precision="16"
        )
    )

    pipeline:Pipeline = config.setup()
    pipeline.process("assets/All the Way North.mp3")

if __name__ == "__main__":
    main_script()       # Run this script with `python dawify/run.py`
    # tyro.cli(main_cli)    # Run this script with `python dawify/run.py --help`