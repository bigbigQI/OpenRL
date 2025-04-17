set -x

   # --use_kl_loss \
   # --use_kl_estimator_k3 \
   # --init_kl_coef 1e-6 \

HDFS_HOME=/apps/OpenRLHF
RUN_NAME=Qwen2.5_1.5B_distill_grpo_14k_reward

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/apps/open"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 2 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.6 \
   --gamma 1.0 \
   --advantage_estimator group_norm \
   --pretrain deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
   --save_path $HDFS_HOME/checkpoints/$RUN_NAME \
   --ckpt_path $HDFS_HOME/checkpoints/$RUN_NAME \
   --save_hf_ckpt \
   --micro_train_batch_size 1 \
   --train_batch_size 1024 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 16 \
   --max_epochs 1 \
   --num_episodes 20 \
   --prompt_max_len 1024 \
   --max_samples 1000000 \
   --generate_max_len 14000 \
   --temperature 0.8 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-6 \
   --prompt_data pe-nlp/ORZ-13K-Hard-NoSys \
   --input_key input \
   --label_key ground_truth_answer \
   --save_steps 1 \
   --flash_attn \
   --init_kl_coef 0.0 \
   --load_checkpoint \
   --use_wandb 149737fd3c4537b349a37aab90b6fff96f385ebc \
   --wandb_run_name $RUN_NAME \
   --wandb_project rl_from_distill_1p5b \
   --max_ckpt_num 5 \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --remote_rm_url /apps/examples/scripts/r1_reward_func.py
