from dawify.post_processing.norm_velocity import VelocityNormalizer


if __name__ == '__main__':
    input_path = '/app/assets/sample_level1_audio/sample_level1_drum.mid'
    output_path = '/app/outputs/tests/sample_level1_audio/sample_level1_drum_norm.mid'

    processor = VelocityNormalizer(input_path)
    processor.process()
    processor.save(output_path)
