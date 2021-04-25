You can lip-sync any video to any audio:
```bash
python inference.py --checkpoint_path <ckpt> --face <video.mp4> --audio <an-audio-source> 
```

##### Preprocess the dataset for fast training

```bash
python preprocess.py --data_root data_root/main --preprocessed_root lrs2_preprocessed/
```

##### Training the Wav2Lip models

```bash
python color_syncnet_train.py --data_root lrs2_preprocessed/ --checkpoint_dir <folder_to_save_checkpoints>
```
```bash
python hq_wav2lip_train.py --data_root lrs2_preprocessed/ --checkpoint_dir <folder_to_save_checkpoints> --syncnet_checkpoint_path <path_to_expert_disc_checkpoint>
```