#! /bin/bash
#
# 【finetune】
#
#  概要:
#      xlnet-japanise を利用してfinetuning する
#
MODEL="/work/model/xlnet"
CMD="./run_classifier.py"
PRETRAINED_MODEL_PATH="${MODEL}/model.ckpt-50000"
FINETUNE_OUTPUT_DIR="${MODEL}/finetune/livedoor_output"
TASK="livedoor"

echo "計算開始: `date`"
(CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ${CMD} \
	 --do_train=true \
	 --do_eval=true \
	 --task_name=${TASK} \
	 --data_dir=/work/data/${TASK} \
	 --output_dir=${FINETUNE_OUTPUT_DIR} \
	 --model_dir=${PRETRAINED_MODEL_PATH} \
         --uncased=False \
	 --spiece_model_file=${MODEL}/wiki-ja-xlnet.model \
	 -model_config_path=${MODEL}/config.json \
	 --init_checkpoint=${PRETRAINED_MODEL_PATH} \
	 --max_seq_length=512 \
	 --train_batch_size=8 \
	 --num_hosts=1 \
	 --num_core_per_host=4 \
	 --learning_rate=1e-7 \
	 --train_steps=1200 \
	 --warmup_steps=120 \
	 --save_steps=600 \
	 --is_regression=True)
echo "計算完了: `date`"
