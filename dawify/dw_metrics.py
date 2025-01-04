import sys
import os.path as osp

mt3_dir = osp.abspath(osp.join(osp.dirname(__file__), 'third_party', 'amt', "src"))
assert osp.exists(mt3_dir), "please modify mt3_dir to point to the amt directory"

if mt3_dir not in sys.path:
    sys.path.insert(0, mt3_dir) 

from dawify.third_party.amt.src.utils.midi import midi2note
from dawify.third_party.amt.src.utils.metrics import compute_track_metrics
from dawify.midify.mt3 import MT3Config
from dawify.mis_utils import rprint, print_metrics

def calc_metric(gt_midi:str, pred_midi:str, model:str):
    config = MT3Config(model_name=model)
    model = config.setup().model

    gt_notes, duration = midi2note(gt_midi, quantize=False, ignore_pedal=True, force_all_program_to=0)
    pred_notes, _ = midi2note(pred_midi, quantize=False, ignore_pedal=True, force_all_program_to=0)

    metrics = compute_track_metrics(pred_notes, 
                                    gt_notes,
                                    eval_vocab=None ,#model.hparams.eval_vocab[0],
                                    eval_drum_vocab=None, #model.hparams.eval_drum_vocab,
                                    onset_tolerance=model.hparams.onset_tolerance,
                                    add_pitch_class_metric=None, #model.hparams.add_pitch_class_metric,
                                    add_melody_metric=['Singing Voice'] if model.hparams.add_melody_metric_to_singing else None,
                                    add_frame_metric=True,
                                    add_micro_metric=True,
                                    add_multi_f_metric=True)
    
    dic = {}
    rprint("[red] Drum needs special treatment, skipping for now [/red]",False)
    for e in metrics[1:]:
        for key in e:
            if key in dic:
                raise ValueError(f"Duplicate key found: {key}")

        dic.update(e)

    return dic

if __name__ == "__main__":
    inp_f = "/home/boss/projects/dawify/assets/sample_level1_audio/sample_level1_audio/sample_level1_drum.mid"
    pred_f = "/home/boss/projects/dawify/outputs/mt3/sample_level1/drums_processed.mid"
    model = "YMT3+"
    metrics = calc_metric(inp_f, pred_f, model)
    print_metrics(metrics)
    assert 0
