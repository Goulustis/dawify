from dawify.post_processing.norm_drum_notes import DrumNoteNormalizer


if __name__ == '__main__':
    input_path = '/app/assets/sample_level1_audio/sample_level1_drum.mid'
    output_path = '/app/outputs/tests/sample_level1_audio/sample_level1_drum_norm_notes.mid'

    processor = DrumNoteNormalizer(input_path)
    processor.process()
    processor.save(output_path)
