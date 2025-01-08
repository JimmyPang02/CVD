#!/bin/bash

# 示例参数
lora_scale=1.0
lora_ckpt_path="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/model/animatediff/v3_sd15_adapter.ckpt"
unet_ckpt_path="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/model/stable-diffusion-v1-5"
save_path="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/model/stable-diffusion-v1-5/unet_webvidlora_v3"
unet_config_path="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/model/stable-diffusion-v1-5/unet/config.json"

# 执行命令
python tools/merge_lora2unet.py --lora_scale=$lora_scale --lora_ckpt_path=$lora_ckpt_path --unet_ckpt_path=$unet_ckpt_path --save_path=$save_path --unet_config_path=$unet_config_path 