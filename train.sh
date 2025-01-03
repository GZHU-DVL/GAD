MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma False"
# shellcheck disable=SC2034
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine --use_kl False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64 --schedule_sampler loss-second-moment"

python scripts/image_train.py --data_dir /x_adv/train --data_gt_dir /x_clean/train  $MODEL_FLAGS $SAMPLE_FLAGS $TRAIN_FLAGS