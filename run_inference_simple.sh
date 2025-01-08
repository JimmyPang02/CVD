#!/bin/bash -l

GPU=$1
SEED=2024
RANDOM_PORT=$((25100 + GPU))

# Parameters: 
# ori_model_path: path to the Stable Diffusion folder (fused with webvid lora)
# pose_adaptor_ckpt: path to the CameraCtrl's pose module checkpoint
# motion_module_ckpt: path to the AnimateDiff's motion module checkpoint
# epi_module_ckpt: path to the trained CVD's module checkpoint
# civitai_base_model (optional): Stable Diffusion's LoRA checkpoint. The webvid LoRA from AnimateDiff will be used if not specified. 
# caption_file: Text prompt file 
# pose_file_0: path to the first pose file
# pose_file_1: path to the second pose file

CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch --nproc_per_node=1 --master_port=${RANDOM_PORT} inference_epi.py \
--out_root ./results/pair_${GPU}/ \
--ori_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/model/stable-diffusion-v1-5 --unet_subfolder unet_webvidlora_v3 \
--pose_adaptor_ckpt /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/model/CVD/CameraCtrl.ckpt \
--motion_module_ckpt /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/model/animatediff/v3_sd15_mm.ckpt \
--epi_module_ckpt /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/model/CVD/cvd.ckpt \
--model_config ./configs/inference_config.yaml \
--caption_file ./assets/cameractrl_prompts.json \
--zero_first_frame_scale \
--image_height 256 \
--image_width 256 \
--no_lora_validation \
--guidance_scale 8.5 \
--pose_adaptor_scale 1.0 \
--global_seed ${SEED} \
--use_negative_prompt \
--num_videos 8 \
--pose_file_0 ./assets/pose_files/2f25826f0d0ef09a.txt \
--pose_file_1 ./assets/pose_files/2c80f9eb0d3b2bb4.txt \

# Other poses options:
# --pose_file_0 /home/shengqu/repos/Epi_CameraCtrl/assets/pose_files/0bf152ef84195293.txt \
# --pose_file_1 /home/shengqu/repos/Epi_CameraCtrl/assets/pose_files/0c9b371cc6225682.txt \

# --pose_file_0 /home/shengqu/repos/Epi_CameraCtrl/assets/pose_files/0bf152ef84195293.txt \
# --pose_file_1 /home/shengqu/repos/Epi_CameraCtrl/assets/pose_files/0bf152ef84195293.txt \

# --pose_file_0 /home/shengqu/repos/Epi_CameraCtrl/assets/pose_files/0bf152ef84195293.txt \
# --pose_file_1 /home/shengqu/repos/Epi_CameraCtrl/assets/pose_files/0c9b371cc6225682.txt \

