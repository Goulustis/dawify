import tyro

from dawify.pipeline.pipeline import PipelineConfig, Pipeline
from dawify.midify.mt3 import MT3Config
from dawify.dw_metrics import calc_and_print_snrs

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

    # inp_dir, out_dir = pipeline.get_in_out_dirs()
    # calc_and_print_snrs(out_dir, inp_dir)

if __name__ == "__main__":
    main_script()       # Run this script with `python dawify/run.py`
    # tyro.cli(main_cli)    # Run this script with `python dawify/run.py --help`