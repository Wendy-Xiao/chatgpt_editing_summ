# ChatGPT-steered Editing Instructor
This is the official code for the paper [ChatGPT-steered Editing Instructor for Customization of Abstractive Summarization]().

## Datasets
We conduct experiments on CNNDM and DeFacto datasets.
* CNNDM: 
    - Dataset with summaries generated by ChatGPT: [here]()
    - Dataset with initial summaries and keyword lists: [here]()
* DeFacto: [here]()

## Step 1: Supervised Training
### Prerequisite: 
Build an environment with `./gpt_iter_env.yaml`.
### Run
Run the supervised training by using `run_instruction_generator_gpt.py` and `run_instruction_generator_gpt_defacto.py`.

Example:
```
python -m torch.distributed.launch --nnodes=1 --nproc_per_node 8 code/run_instruction_generator_gpt.py \
          --output_dir ./instruction_generator_gpt_flant5 \
          --do_train \
          --per_device_train_batch_size 8 \
          --gradient_accumulation_steps 4 \
          --do_eval \
          --per_device_eval_batch_size 8 \
          --evaluation_strategy steps \
          --predict_with_generate \
          --eval_steps 5000 \
          --num_train_epochs 10 \
          --save_strategy steps \
          --save_steps 5000 \
          --save_total_limit 10 \
          --warmup_ratio 0.05 \
          --logging_steps 100 \
          --dataset_path PATH/TO/DATASET \
          --dataset_name cnndm_gptsumm_with_kw \
```

## Step 2: Editor-steered Reinforcement Learning
### Prerequisite
To run this step, please build an environment following the instructions in `./RL4LMs/README.md`.
### Run
We use the package RL4LMs for reinforcement learning. To make the package work well with our model, we modified the code of the package, and add customized reward `./RL4LMs/rl4lms/envs/text_generation/custom_reward.py` and metric `./RL4LMs/rl4lms/envs/text_generation/custom_metric.py`, as well as the customized data loader `./RL4LMs\rl4lms\data_pools\custom_pool.py`.

To run the code, modify the config files under `./rl_config`, and call
```
 python train_text_generation.py --config_path ./rl_config/instruction_chatgpt_nlpo_t5flan_kw_reward.yml --experiment_name chatgpt_nlpo_cnndm

```

Note that the checkpoints saved by the RL process are different from the checkpoints used in Huggingface Trainer. To predict with the models trained by RL, you need to first extract and save the model by `./utils.ipynb`
## Evaluation
To evaluate the models, you need to generate the edited summaries.
* First predict the instructions 
```
python code/run_instruction_generator_gpt.py \
          --output_dir ./output/ \
          --do_predict \
          --per_device_eval_batch_size 16 \
          --predict_with_generate \
          --load_from_checkpoint \
          --ckpt_path PATH/TO/MODEL/CHECKPOINTS \
          --dataset_path ../data/cnndm_gptsumm_chatgpt \
          --dataset_name cnndm_chatgptsumm_with_kw \
          --pretrained_cache_path ../data/ \
          --outfile_name OUTPUT_NAME
```

* Then generate summaries with the generated instructions by `run_summary_generator_with_instruction.py` and `run_summary_generator_with_instruction_defacto.py`. Replace the place holders to the correct path.


### Evaluation Metrics
For CNNDM, directly run `evaluate_cnndm.py` with the generated summaries.
For DeFacto, checkout [DAC](https://github.com/tagoyal/factuality-datasets) and [QFE](https://github.com/salesforce/QAFactEval)