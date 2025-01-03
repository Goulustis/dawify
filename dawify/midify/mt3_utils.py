import sys
import os.path as osp

mt3_dir = osp.abspath(osp.join(osp.dirname(__file__), '..', 'third_party', 'amt', "src"))
if mt3_dir not in sys.path:
    sys.path.insert(0, mt3_dir) 

from dawify.third_party.amt.src.model.init_train import update_config


from collections import Counter
import argparse
import torch
import torchaudio
import os
from typing import Tuple, Literal, Dict

from copy import deepcopy
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from dawify.third_party.amt.src.config.config import shared_cfg as default_shared_cfg
from dawify.third_party.amt.src.config.config import DEEPSPEED_CFG
from dawify.third_party.amt.src.model.init_train import update_config
from dawify.third_party.amt.src.utils.task_manager import TaskManager
from dawify.third_party.amt.src.config.vocabulary import drum_vocab_presets
from dawify.third_party.amt.src.utils.utils import str2bool
from dawify.third_party.amt.src.utils.utils import Timer
from dawify.third_party.amt.src.utils.audio import slice_padded_array
from dawify.third_party.amt.src.utils.note2event import mix_notes
from dawify.third_party.amt.src.utils.event2note import merge_zipped_note_events_and_ties_to_notes
from dawify.third_party.amt.src.utils.utils import write_model_output_as_midi
from dawify.third_party.amt.src.model.ymt3 import YourMT3

#NOTE: This is a modified version of the original function from amt/src/model/init_train.py
def initialize_trainer(args: argparse.Namespace,
                       stage: Literal['train', 'test'] = 'train') -> Tuple[pl.Trainer, WandbLogger, dict]:
    """Initialize trainer and logger"""
    shared_cfg = deepcopy(default_shared_cfg)

    # create save dir
    # os.makedirs(shared_cfg["WANDB"]["save_dir"], exist_ok=True)

    # collecting specific checkpoint from exp_id with extension (@xxx where xxx is checkpoint name)
    if "@" in args.exp_id:
        args.exp_id, checkpoint_name = args.exp_id.split("@")
    else:
        checkpoint_name = "last.ckpt"

    # checkpoint dir
    save_dir = osp.join(os.path.abspath(os.path.join(__file__, "..", "..")), "third_party", "amt", "logs")
    lightning_dir = os.path.join(save_dir, args.project, args.exp_id)

    # create logger
    if args.wandb_mode is not None:
        shared_cfg["WANDB"]["mode"] = str(args.wandb_mode)
    if shared_cfg["WANDB"].get("cache_dir", None) is not None:
        os.environ["WANDB_CACHE_DIR"] = shared_cfg["WANDB"].get("cache_dir")
        del shared_cfg["WANDB"]["cache_dir"]  # remove cache_dir from shared_cfg
    wandb_logger = WandbLogger(log_model="all",
                               project=args.project,
                               id=args.exp_id,
                               allow_val_change=True,
                               **shared_cfg['WANDB'])

    # check if any checkpoint exists
    last_ckpt_path = os.path.join(lightning_dir, "checkpoints", checkpoint_name)
    if os.path.exists(os.path.join(last_ckpt_path)):
        print(f'Resuming from {last_ckpt_path}')
    elif stage == 'train':
        print(f'No checkpoint found in {last_ckpt_path}. Starting from scratch')
        last_ckpt_path = None
    else:
        raise ValueError(f'No checkpoint found in {last_ckpt_path}. Quit...')

    # add info
    dir_info = dict(lightning_dir=lightning_dir, last_ckpt_path=last_ckpt_path)

    # define checkpoint callback
    checkpoint_callback = ModelCheckpoint(**shared_cfg["CHECKPOINT"],)

    # define lr scheduler monitor callback
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # deepspeed strategy
    if args.strategy == 'deepspeed':
        strategy = pl.strategies.DeepSpeedStrategy(config=DEEPSPEED_CFG)

    # validation interval
    if stage == 'train' and args.val_interval is not None:
        shared_cfg["TRAINER"]["check_val_every_n_epoch"] = None
        shared_cfg["TRAINER"]["val_check_interval"] = int(args.val_interval)

    # define trainer
    sync_batchnorm = False
    if stage == 'train':
        # train batch size
        if args.train_batch_size is not None:
            train_sub_bsz = int(args.train_batch_size[0])
            train_local_bsz = int(args.train_batch_size[1])
            if train_local_bsz % train_sub_bsz == 0:
                shared_cfg["BSZ"]["train_sub"] = train_sub_bsz
                shared_cfg["BSZ"]["train_local"] = train_local_bsz
            else:
                raise ValueError(
                    f'Local batch size {train_local_bsz} must be divisible by sub batch size {train_sub_bsz}')

        # ddp strategy
        if args.strategy == 'ddp':
            args.strategy = 'ddp_find_unused_parameters_true'  # fix for conformer or pitchshifter having unused parameter issue

            # sync-batchnorm
            if args.sync_batchnorm is True:
                sync_batchnorm = True

    train_params = dict(**shared_cfg["TRAINER"],
                        devices=args.num_gpus if args.num_gpus == 'auto' else int(args.num_gpus),
                        num_nodes=int(args.num_nodes),
                        strategy=strategy if args.strategy == 'deepspeed' else args.strategy,
                        precision=args.precision,
                        max_epochs=args.max_epochs if stage == 'train' else None,
                        max_steps=args.max_steps if stage == 'train' else -1,
                        logger=wandb_logger,
                        callbacks=[checkpoint_callback, lr_monitor],
                        sync_batchnorm=sync_batchnorm)
    trainer = pl.trainer.trainer.Trainer(**train_params)

    # Update wandb logger (for DDP)
    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update(args, allow_val_change=True)

    return trainer, wandb_logger, dir_info, shared_cfg

def load_model_checkpoint(args=None):
    parser = argparse.ArgumentParser(description="YourMT3")
    # General
    parser.add_argument('exp_id', type=str, help='A unique identifier for the experiment is used to resume training. The "@" symbol can be used to load a specific checkpoint.')
    parser.add_argument('-p', '--project', type=str, default='ymt3', help='project name')
    parser.add_argument('-ac', '--audio-codec', type=str, default=None, help='audio codec (default=None). {"spec", "melspec"}. If None, default value defined in config.py will be used.')
    parser.add_argument('-hop', '--hop-length', type=int, default=None, help='hop length in frames (default=None). {128, 300} 128 for MT3, 300 for PerceiverTFIf None, default value defined in config.py will be used.')
    parser.add_argument('-nmel', '--n-mels', type=int, default=None, help='number of mel bins (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-if', '--input-frames', type=int, default=None, help='number of audio frames for input segment (default=None). If None, default value defined in config.py will be used.')
    # Model configurations
    parser.add_argument('-sqr', '--sca-use-query-residual', type=str2bool, default=None, help='sca use query residual flag. Default follows config.py')
    parser.add_argument('-enc', '--encoder-type', type=str, default=None, help="Encoder type. 't5' or 'perceiver-tf' or 'conformer'. Default is 't5', following config.py.")
    parser.add_argument('-dec', '--decoder-type', type=str, default=None, help="Decoder type. 't5' or 'multi-t5'. Default is 't5', following config.py.")
    parser.add_argument('-preenc', '--pre-encoder-type', type=str, default='default', help="Pre-encoder type. None or 'conv' or 'default'. By default, t5_enc:None, perceiver_tf_enc:conv, conformer:None")
    parser.add_argument('-predec', '--pre-decoder-type', type=str, default='default', help="Pre-decoder type. {None, 'linear', 'conv1', 'mlp', 'group_linear'} or 'default'. Default is {'t5': None, 'perceiver-tf': 'linear', 'conformer': None}.")
    parser.add_argument('-cout', '--conv-out-channels', type=int, default=None, help='Number of filters for pre-encoder conv layer. Default follows "model_cfg" of config.py.')
    parser.add_argument('-tenc', '--task-cond-encoder', type=str2bool, default=True, help='task conditional encoder (default=True). True or False')
    parser.add_argument('-tdec', '--task-cond-decoder', type=str2bool, default=True, help='task conditional decoder (default=True). True or False')
    parser.add_argument('-df', '--d-feat', type=int, default=None, help='Audio feature will be projected to this dimension for Q,K,V of T5 or K,V of Perceiver (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-pt', '--pretrained', type=str2bool, default=False, help='pretrained T5(default=False). True or False')
    parser.add_argument('-b', '--base-name', type=str, default="google/t5-v1_1-xxl", help='base model name (default="google/t5-v1_1-xxl")')
    parser.add_argument('-epe', '--encoder-position-encoding-type', type=str, default='default', help="Positional encoding type of encoder. By default, pre-defined PE for T5 or Perceiver-TF encoder in config.py. For T5: {'sinusoidal', 'trainable'}, conformer: {'rotary', 'trainable'}, Perceiver-TF: {'trainable', 'rope', 'alibi', 'alibit', 'None', '0', 'none', 'tkd', 'td', 'tk', 'kdt'}.")
    parser.add_argument('-dpe', '--decoder-position-encoding-type', type=str, default='default', help="Positional encoding type of decoder. By default, pre-defined PE for T5 in config.py. {'sinusoidal', 'trainable'}.")
    parser.add_argument('-twe', '--tie-word-embedding', type=str2bool, default=None, help='tie word embedding (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-el', '--event-length', type=int, default=None, help='event length (default=None). If None, default value defined in model cfg of config.py will be used.')
    # Perceiver-TF configurations
    parser.add_argument('-dl', '--d-latent', type=int, default=None, help='Latent dimension of Perceiver. On T5, this will be ignored (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-nl', '--num-latents', type=int, default=None, help='Number of latents of Perceiver. On T5, this will be ignored (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-dpm', '--perceiver-tf-d-model', type=int, default=None, help='Perceiver-TF d_model (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-npb', '--num-perceiver-tf-blocks', type=int, default=None, help='Number of blocks of Perceiver-TF. On T5, this will be ignored (default=None). If None, default value defined in config.py.')
    parser.add_argument('-npl', '--num-perceiver-tf-local-transformers-per-block', type=int, default=None, help='Number of local layers per block of Perceiver-TF. On T5, this will be ignored (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-npt', '--num-perceiver-tf-temporal-transformers-per-block', type=int, default=None, help='Number of temporal layers per block of Perceiver-TF. On T5, this will be ignored (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-atc', '--attention-to-channel', type=str2bool, default=None, help='Attention to channel flag of Perceiver-TF. On T5, this will be ignored (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-ln', '--layer-norm-type', type=str, default=None, help='Layer normalization type (default=None). {"layer_norm", "rms_norm"}. If None, default value defined in config.py will be used.')
    parser.add_argument('-ff', '--ff-layer-type', type=str, default=None, help='Feed forward layer type (default=None). {"mlp", "moe", "gmlp"}. If None, default value defined in config.py will be used.')
    parser.add_argument('-wf', '--ff-widening-factor', type=int, default=None, help='Feed forward layer widening factor for MLP/MoE/gMLP (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-nmoe', '--moe-num-experts', type=int, default=None, help='Number of experts for MoE (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-kmoe', '--moe-topk', type=int, default=None, help='Top-k for MoE (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-act', '--hidden-act', type=str, default=None, help='Hidden activation function (default=None). {"gelu", "silu", "relu", "tanh"}. If None, default value defined in config.py will be used.')
    parser.add_argument('-rt', '--rotary-type', type=str, default=None, help='Rotary embedding type expressed in three letters. e.g. ppl: "pixel" for SCA and latents, "lang" for temporal transformer. If None, use config.')
    parser.add_argument('-rk', '--rope-apply-to-keys', type=str2bool, default=None, help='Apply rope to keys (default=None). If None, use config.')
    parser.add_argument('-rp', '--rope-partial-pe', type=str2bool, default=None, help='Whether to apply RoPE to partial positions (default=None). If None, use config.')
    # Decoder configurations
    parser.add_argument('-dff', '--decoder-ff-layer-type', type=str, default=None, help='Feed forward layer type of decoder (default=None). {"mlp", "moe", "gmlp"}. If None, default value defined in config.py will be used.')
    parser.add_argument('-dwf', '--decoder-ff-widening-factor', type=int, default=None, help='Feed forward layer widening factor for decoder MLP/MoE/gMLP (default=None). If None, default value defined in config.py will be used.')
    # Task and Evaluation configurations
    parser.add_argument('-tk', '--task', type=str, default='mt3_full_plus', help='tokenizer type (default=mt3_full_plus). See config/task.py for more options.')
    parser.add_argument('-epv', '--eval-program-vocab', type=str, default=None, help='evaluation vocabulary (default=None). If None, default vocabulary of the data preset will be used.')
    parser.add_argument('-edv', '--eval-drum-vocab', type=str, default=None, help='evaluation vocabulary for drum (default=None). If None, default vocabulary of the data preset will be used.')
    parser.add_argument('-etk', '--eval-subtask-key', type=str, default='default', help='evaluation subtask key (default=default). See config/task.py for more options.')
    parser.add_argument('-t', '--onset-tolerance', type=float, default=0.05, help='onset tolerance (default=0.05).')
    parser.add_argument('-os', '--test-octave-shift', type=str2bool, default=False, help='test optimal octave shift (default=False). True or False')
    parser.add_argument('-w', '--write-model-output', type=str2bool, default=True, help='write model test output to file (default=False). True or False')
    # Trainer configurations
    parser.add_argument('-pr','--precision', type=str, default="bf16-mixed", help='precision (default="bf16-mixed") {32, 16, bf16, bf16-mixed}')
    parser.add_argument('-st', '--strategy', type=str, default='auto', help='strategy (default=auto). auto or deepspeed or ddp')
    parser.add_argument('-n', '--num-nodes', type=int, default=1, help='number of nodes (default=1)')
    parser.add_argument('-g', '--num-gpus', type=str, default='auto', help='number of gpus (default="auto")')
    parser.add_argument('-wb', '--wandb-mode', type=str, default="disabled", help='wandb mode for logging (default=None). "disabled" or "online" or "offline". If None, default value defined in config.py will be used.')
    # Debug
    parser.add_argument('-debug', '--debug-mode', type=str2bool, default=False, help='debug mode (default=False). True or False')
    parser.add_argument('-tps', '--test-pitch-shift', type=int, default=None, help='use pitch shift when testing. debug-purpose only. (default=None). semitone in int.')
    args = parser.parse_args(args)
    # yapf: enable
    if torch.__version__ >= "1.13":
        torch.set_float32_matmul_precision("high")
    args.epochs = None

    # Initialize and update config
    _, _, dir_info, shared_cfg = initialize_trainer(args, stage='test')
    shared_cfg, audio_cfg, model_cfg = update_config(args, shared_cfg, stage='test')

    if args.eval_drum_vocab != None:  # override eval_drum_vocab
        eval_drum_vocab = drum_vocab_presets[args.eval_drum_vocab]

    # Initialize task manager
    tm = TaskManager(task_name=args.task,
                     max_shift_steps=int(shared_cfg["TOKENIZER"]["max_shift_steps"]),
                     debug_mode=args.debug_mode)
    print(f"Task: {tm.task_name}, Max Shift Steps: {tm.max_shift_steps}")

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = YourMT3(
        audio_cfg=audio_cfg,
        model_cfg=model_cfg,
        shared_cfg=shared_cfg,
        optimizer=None,
        task_manager=tm,  # tokenizer is a member of task_manager
        eval_subtask_key=args.eval_subtask_key,
        write_output_dir=dir_info["lightning_dir"] if args.write_model_output or args.test_octave_shift else None
        ).to(device)
    checkpoint = torch.load(dir_info["last_ckpt_path"])
    state_dict = checkpoint['state_dict']
    new_state_dict = {k: v for k, v in state_dict.items() if 'pitchshift' not in k}
    model.load_state_dict(new_state_dict, strict=False)
    return model.eval()


def transcribe(model, audio_info, out_dir="outputs/mt3"):
    t = Timer()

    # Converting Audio
    t.start()
    audio, sr = torchaudio.load(uri=audio_info['filepath'])
    audio = torch.mean(audio, dim=0).unsqueeze(0)
    audio = torchaudio.functional.resample(audio, sr, model.audio_cfg['sample_rate'])
    audio_segments = slice_padded_array(audio, model.audio_cfg['input_frames'], model.audio_cfg['input_frames'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_segments = torch.from_numpy(audio_segments.astype('float32')).to(device).unsqueeze(1) # (n_seg, 1, seg_sz)
    t.stop(); t.print_elapsed_time("converting audio");

    # Inference
    t.start()
    pred_token_arr, _ = model.inference_file(bsz=8, audio_segments=audio_segments)
    t.stop(); t.print_elapsed_time("model inference");

    # Post-processing
    t.start()
    num_channels = model.task_manager.num_decoding_channels
    n_items = audio_segments.shape[0]
    start_secs_file = [model.audio_cfg['input_frames'] * i / model.audio_cfg['sample_rate'] for i in range(n_items)]
    pred_notes_in_file = []
    n_err_cnt = Counter()
    for ch in range(num_channels):
        pred_token_arr_ch = [arr[:, ch, :] for arr in pred_token_arr]  # (B, L)
        zipped_note_events_and_tie, list_events, ne_err_cnt = model.task_manager.detokenize_list_batches(
            pred_token_arr_ch, start_secs_file, return_events=True)
        pred_notes_ch, n_err_cnt_ch = merge_zipped_note_events_and_ties_to_notes(zipped_note_events_and_tie)
        pred_notes_in_file.append(pred_notes_ch)
        n_err_cnt += n_err_cnt_ch
    pred_notes = mix_notes(pred_notes_in_file)  # This is the mixed notes from all channels

    # Write MIDI
    write_model_output_as_midi(pred_notes, out_dir,
                              audio_info['track_name'], model.midi_output_inverse_vocab, output_dir_suffix="..")
    t.stop(); t.print_elapsed_time("post processing")
    midifile =  os.path.join(out_dir, audio_info['track_name']  + '.mid')
    assert os.path.exists(midifile)
    return midifile


def load_model(model_name = 'YPTF+Single (noPS)', precision = '16', project = '2024'):
    # model_name = 'YPTF+Single (noPS)' # @param ["YMT3+", "YPTF+Single (noPS)", "YPTF+Multi (PS)", "YPTF.MoE+Multi (noPS)", "YPTF.MoE+Multi (PS)"]
    # precision = '16' # @param ["32", "bf16-mixed", "16"]
    # project = '2024'

    if model_name == "YMT3+":
        checkpoint = "notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72@model.ckpt"
        args = [checkpoint, '-p', project, '-pr', precision]
    elif model_name == "YPTF+Single (noPS)":
        checkpoint = "ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100@model.ckpt"
        args = [checkpoint, '-p', project, '-enc', 'perceiver-tf', '-ac', 'spec',
                '-hop', '300', '-atc', '1', '-pr', precision]
    elif model_name == "YPTF+Multi (PS)":
        checkpoint = "mc13_256_all_cross_v6_xk5_amp0811_edr005_attend_c_full_plus_2psn_nl26_sb_b26r_800k@model.ckpt"
        args = [checkpoint, '-p', project, '-tk', 'mc13_full_plus_256',
                '-dec', 'multi-t5', '-nl', '26', '-enc', 'perceiver-tf',
                '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', precision]
    elif model_name == "YPTF.MoE+Multi (noPS)":
        checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
        args = [checkpoint, '-p', project, '-tk', 'mc13_full_plus_256', '-dec', 'multi-t5',
                '-nl', '26', '-enc', 'perceiver-tf', '-sqr', '1', '-ff', 'moe',
                '-wf', '4', '-nmoe', '8', '-kmoe', '2', '-act', 'silu', '-epe', 'rope',
                '-rp', '1', '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', precision]
    elif model_name == "YPTF.MoE+Multi (PS)":
        checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2@model.ckpt"
        args = [checkpoint, '-p', project, '-tk', 'mc13_full_plus_256', '-dec', 'multi-t5',
                '-nl', '26', '-enc', 'perceiver-tf', '-sqr', '1', '-ff', 'moe',
                '-wf', '4', '-nmoe', '8', '-kmoe', '2', '-act', 'silu', '-epe', 'rope',
                '-rp', '1', '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', precision]
    else:
        raise ValueError(model_name)

    model = load_model_checkpoint(args=args)
    return model


def prepare_media(source_path_or_url: os.PathLike) -> Dict:
    """prepare media from source path or youtube, and return audio info"""
    # Get audio_file
    source_type = 'audio_filepath'
    if source_type == 'audio_filepath':
        audio_file = source_path_or_url

    # Create info
    info = torchaudio.info(audio_file)
    return {
        "filepath": audio_file,
        "track_name": os.path.basename(audio_file).split('.')[0],
        "sample_rate": int(info.sample_rate),
        "bits_per_sample": int(info.bits_per_sample),
        "num_channels": int(info.num_channels),
        "num_frames": int(info.num_frames),
        "duration": int(info.num_frames / info.sample_rate),
        "encoding": str.lower(info.encoding),
        }


def process_audio(model, audio_filepath, out_dir="outputs/mt3"):
    if audio_filepath is None:
        return None
    audio_info = prepare_media(audio_filepath)
    midifile = transcribe(model, audio_info, out_dir)
    return midifile
    # midifile = to_data_url(midifile)
    # return create_html_from_midi(midifile) # html midiplayer