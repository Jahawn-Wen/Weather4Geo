python train_1126_stage2_3_more.py \
--name='qwen11.24_xvlm_focal_reptile_32B_spade_ft_fgclip2_1201_inv_scale_stage2_3_moreA2' \
--experiment_name='qwen11.24_xvlm_focal_reptile_32B_spade_ft_fgclip2_1201_inv_scale_stage2_3_moreA2' \
--data_dir='${TRAIN_DIR}' \
--views=3 \
--droprate=0.5 \
--extra \
--share \
--stride=1 \
--h=384 \
--w=384 \
--lr=0.005 \
--gpu_ids='1' \
--norm='spade' \
--reptile \
--qwen \
--iaa \
--focal \
--multi_weather \
--btnk 0 1 1 0 0 0 0 \
--conv_norm='none' \
--adain='a' \
--lambda_scene 0.10 \
--lambda_weather 0.03 \
--lambda_weather_inv 0.01 \
--stage2_start_epoch 180 \
--stage2_ramp_epochs 20 \
--proto_m 0.05 \
--moving_avg 0.995








