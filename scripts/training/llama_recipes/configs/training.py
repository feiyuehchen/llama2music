# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str="/home/feiyuehchen/personality/llama/llama-2-7b-hf"
    # tokenizer_dir: str="/home/feiyuehchen/personality/llama2music/tokenizer/piano_planC"
    tokenizer_dir: str="/home/feiyuehchen/personality/llama2music/tokenizer/planB"
    # tokenizer_dir: str="/home/feiyuehchen/personality/llama/llama-2-7b-hf"
    # tokenizer_dir: str="/home/feiyuehchen/personality/llama2music/tokenizer/ailab17k_data/planC"
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    batch_size_training: int=2
    batching_strategy: str="packing" #alternative: padding
    context_length: int=2048
    gradient_accumulation_steps: int=8
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
    num_epochs: int=10
    num_workers_dataloader: int=32
    lr: float=5e-4
    weight_decay: float=0.05
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=False
    val_batch_size: int=2
    dataset = "ailab_dataset"
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=False
    use_pretrained_peft_model: bool=False
    peft_model: str="PATH/to/save/PEFT/3_ailab_planC_LR2e-4" #pretrained peft model
    output_dir: str = "PATH/to/save/PEFT/ailab_planB_seg"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=True # will be used if using FSDP
    use_fast_kernels: bool = True # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    save_metrics: bool = True # saves training metrics to a json file for later plotting
