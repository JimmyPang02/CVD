GPU=$1
CAMERA_TYPE=$2
VIEW_NUM=$3
MASTER_PORT=$(expr 27000 + $GPU)

# Parameters: 
# ori_model_path: path to the Stable Diffusion folder (fused with webvid lora)
# pose_adaptor_ckpt: path to the CameraCtrl's pose module checkpoint
# motion_module_ckpt: path to the AnimateDiff's motion module checkpoint
# epi_module_ckpt: path to the trained CVD's module checkpoint
# civitai_base_model (optional): Stable Diffusion's LoRA checkpoint. The webvid LoRA from AnimateDiff will be used if not specified. 
# caption_file: Text prompt file 
# view_num: Number of generated multi-view videos 
# multistep: Number of recurrent steps for each denoising step
# multiseed: Number of samples for each text prompt
# accumulate_step: Number of pairs assigned to each video (default: 1)
# cam_pattern: pattern of camera trajectories (supported inputs: circle, interpolate, upper_hemi)

if [ $CAMERA_TYPE == "circle" ]; then
    CAMERA_CONFIG='--caption_file assets/cameractrl_prompts_for_circle.json --cam_pattern circle'
elif [ $CAMERA_TYPE == "interpolate" ]; then
    CAMERA_CONFIG='--caption_file assets/cameractrl_prompts_for_interpolate.json --cam_pattern interpolate'
else
    echo "Invalid camera trajectory"
    exit 1
fi

if [ $VIEW_NUM == "4" ]; then
    VIDEO_CONFIG='--view_num 4 --multistep 3'
elif [ $VIEW_NUM == "6" ]; then
    VIDEO_CONFIG='--view_num 6 --multistep 6 --accumulate_step 2'
else
    echo "Invalid video number"
    exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=1 --master_port=$MASTER_PORT inference_epi_advanced.py \
--out_root results/${CAMERA_TYPE}_$GPU \
--ori_model_path ./models/StableDiffusion --unet_subfolder unet_webvidlora_v3 \
--pose_adaptor_ckpt ./models/CameraCtrl.ckpt \
--motion_module_ckpt ./models/animatediff_mm.ckpt \
--epi_module_ckpt ./models/CVD.ckpt \
--model_config ./configs/inference_config.yaml \
--use_specific_seeds --zero_first_frame_scale \
--image_height 256 \
--image_width 256 \
--num_inference_steps 25 \
--multiseed 3 \
$CAMERA_CONFIG $VIDEO_CONFIG \
--civitai_base_model ./models/realisticVisionV60B1_v51VAE.safetensors \