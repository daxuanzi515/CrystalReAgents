from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import torch
import transformers
from struq import SupervisedDataset
from config import IGNORE_INDEX, DEFAULT_TOKENS, SPECIAL_DELM_TOKENS, TEXTUAL_DELM_TOKENS

# MODIFIED: 新增LoRA和bitsandbytes导入与4bit量化支持
from peft import LoraConfig, get_peft_model, TaskType
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig  # 新增：4bit量化配置

@dataclass
class ModelArguments: 
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    window_size: int = field(default=0, metadata={"help": "Window size for the sliding window attention."})
    padding_side: str = field(default="right", metadata={"help": "Padding side for tokenization."})

@dataclass
class DataArguments: 
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    max_samples: Optional[int] = field(default=None, metadata={"help": "Max number of samples to use for training"})

@dataclass
class AttackArguments: 
    attack: str = field(default='alpaca', metadata={"help": "Attack type."})

# MODIFIED: 在TrainingArguments中添加4bit量化相关参数和LoRA参数
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = None
    optim: str = field(default="adamw_torch")
    model_max_length: int = 512  # 不用 field，直接赋值
    downsample: Optional[bool] = True
    lr_scale: Optional[bool] = True

    # 量化参数
    load_in_8bit: bool = field(default=False, metadata={"help": "Enable 8-bit quantization"})
    load_in_4bit: bool = field(default=False, metadata={"help": "Enable 4-bit quantization"})
    bnb_4bit_use_double_quant: bool = field(default=True, metadata={"help": "bnb 4bit double quantization"})
    bnb_4bit_quant_type: str = field(default="nf4", metadata={"help": "bnb 4bit quant type, e.g., nf4 or fp4"})
    bnb_4bit_compute_dtype: Optional[str] = field(default="float16", metadata={"help": "compute dtype for 4bit quantization: float16 or bfloat16"})

    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
    
    evaluation_strategy: str = "no"
    save_strategy: str = "no"
    save_total_limit: int = 1
    report_to: str = field(default="none", metadata={"help": "Disable reporting like wandb"})

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def get_embedding_indices(tokenizer):
    init_values = [tokenizer.encode(v, add_special_tokens=False)[0] for v in TEXTUAL_DELM_TOKENS]
    ignore_values = [i for i in range(len(tokenizer)) if tokenizer.decode(i) == "#"]
    return init_values, ignore_values

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    assert len(SPECIAL_DELM_TOKENS) == len(TEXTUAL_DELM_TOKENS)
    num_new_tokens = tokenizer.add_special_tokens({
        'pad_token': DEFAULT_TOKENS['pad_token'],
        'additional_special_tokens': SPECIAL_DELM_TOKENS
        })
    model.resize_token_embeddings(len(tokenizer))
    delimiter_init_embed_index_from_text = [tokenizer.encode(v, add_special_tokens=False)[0] for v in TEXTUAL_DELM_TOKENS]
    assert num_new_tokens == len(SPECIAL_DELM_TOKENS) + 1

    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data

    # Initialize the [PAD] token with the mean of all embeddings
    input_embeddings[-num_new_tokens] = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    output_embeddings[-num_new_tokens] = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

    # Initialize special delimiters
    for i in range(len(SPECIAL_DELM_TOKENS)):
        index = -num_new_tokens + i + 1
        print('Initialize special delimiter token', tokenizer.decode(len(tokenizer) + index), 'from the embedding of', tokenizer.decode(delimiter_init_embed_index_from_text[i]))
        input_embeddings[index] = input_embeddings[delimiter_init_embed_index_from_text[i]]
        output_embeddings[index] = output_embeddings[delimiter_init_embed_index_from_text[i]]

def make_supervised_data_module(tokenizer, data_args, downsample=True):
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        attack=data_args.attack,
        downsample=downsample,
        max_samples=data_args.max_samples
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, AttackArguments))
    model_args, data_args, training_args, attack_args = parser.parse_args_into_dataclasses()
    data_args.attack = attack_args.attack
    
    # traing_args.output_dir = training_args.output_dir + '_' + data_args.attack
    print('\n\n' + training_args.output_dir + '\n\n')
    print(model_args.model_name_or_path)

    # 读取量化和LoRA参数
    load_in_8bit = training_args.load_in_8bit
    load_in_4bit = training_args.load_in_4bit
    bnb_4bit_config = None
    if load_in_4bit:
        compute_dtype = torch.float16 if training_args.bnb_4bit_compute_dtype.lower() == 'float16' else torch.bfloat16
        bnb_4bit_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=training_args.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=training_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype
        )

    lora_r = training_args.lora_r
    lora_alpha = training_args.lora_alpha
    lora_dropout = training_args.lora_dropout

    # 加载模型，优先使用4bit量化配置
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        quantization_config=bnb_4bit_config if load_in_4bit else None,
        load_in_8bit=load_in_8bit if not load_in_4bit else False,  # 二选一
        device_map="auto" if (load_in_8bit or load_in_4bit) else None,
    )

    if model_args.window_size > 0:
        model.config.window = model_args.window_size

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
    )

    special_tokens_dict = dict()
    special_tokens_dict["pad_token"] = DEFAULT_TOKENS['pad_token']
    special_tokens_dict["eos_token"] = DEFAULT_TOKENS['eos_token']
    special_tokens_dict["bos_token"] = DEFAULT_TOKENS['bos_token']
    special_tokens_dict["unk_token"] = DEFAULT_TOKENS['unk_token']
    special_tokens_dict["additional_special_tokens"] = SPECIAL_DELM_TOKENS
    smart_tokenizer_and_embedding_resize(special_tokens_dict=special_tokens_dict, tokenizer=tokenizer, model=model)

    # LoRA 只在量化启用时才加载
    if load_in_8bit or load_in_4bit:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        # 冻结非LoRA参数，只训练LoRA部分
        for name, param in model.named_parameters():
            if not getattr(param, "requires_grad", False):
                param.requires_grad = False

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, downsample=training_args.downsample)
    
    if not training_args.downsample and training_args.lr_scale:
        training_args.learning_rate /= data_module["train_dataset"].data_copy_count

    trainer = transformers.Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()

