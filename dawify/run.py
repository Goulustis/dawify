
from dawify.pipeline.pipeline import PipelineConfig, Pipeline
from dawify.midify.mt3 import MT3Config

def main():
    config = PipelineConfig(
        mt3_config=MT3Config(
            model_name="YMT3+",
            precision="16"
        )
    )

    pipeline:Pipeline = config.setup()
    pipeline.process("path/to/mp3")

if __name__ == "__main__":
    main()