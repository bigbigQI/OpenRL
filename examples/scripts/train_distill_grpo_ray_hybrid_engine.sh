set -x


export VLLM_FLASH_ATTN_VERSION=2
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_NVLS_ENABLE=0

   # --use_kl_loss \
   # --use_kl_estimator_k3 \
   # --init_kl_coef 1e-6 \

HDFS_HOME=/apps/OpenRLHF
RUN_NAME=Qwen2.5_7B_distill_grpo_10k
# RAY_ADDRESS='http://127.0.0.1:8265' ray job submit --working-dir . -- python my_script.py
RAY_ADDRESS='http://127.0.0.1:8265' ray job submit \
   --runtime-env-json='{
   	"working_dir": "/apps/open",
	"env_vars": {
            "VLLM_FLASH_ATTN_VERSION": "2",
            "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "NCCL_NVLS_ENABLE": "0"
        }
   }' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 4 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 4 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 1 \
   --vllm_gpu_memory_utilization 0.6 \
   --colocate_all_models \
   --gamma 1.0 \
   --advantage_estimator group_norm \
   --pretrain deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
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
   --generate_max_len 10240 \
   --temperature 0.8 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 2e-6 \
   --prompt_data pe-nlp/Skywork-OR1-noSys \
   --input_key input \
   --label_key ground_truth_answer \
   --save_steps 5 \
   --flash_attn \
   --init_kl_coef 0.0 \
   --load_checkpoint \
   --use_wandb 149737fd3c4537b349a37aab90b6fff96f385ebc \
   --wandb_run_name $RUN_NAME \
   --wandb_project rl_from_distill_7b_b200 \
   --max_ckpt_num 5 \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --remote_rm_url /apps/examples/scripts/r1_reward_func.py
