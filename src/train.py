import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer
import os

def train():
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    # Alternatively use a smaller model for faster iteration if needed
    # model_id = "Qwen/Qwen2.5-3B-Instruct"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # BitsAndBytes configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # model = prepare_model_for_kbit_training(model) # SFTTrainer handles this
    
    # LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # model = get_peft_model(model, peft_config) # SFTTrainer handles this if we pass peft_config
    
    # Load dataset
    dataset = load_dataset("json", data_files="datasets/train_refusal.json", split="train")
    
    def formatting_prompts_func(example):
        text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{example['instruction']} {example['input']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
        return text

    from trl import SFTConfig
    
    # Training arguments
    training_args = SFTConfig(
        output_dir="results/finetuned_model",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        num_train_epochs=3, # Fewer epochs for more nuance
        save_steps=10,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        push_to_hub=False,
        report_to="none",
        max_length=512,
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        # max_seq_length is now in training_args (SFTConfig)
        processing_class=tokenizer,
        formatting_func=formatting_prompts_func,
        args=training_args,
    )
    
    trainer.train()
    
    # Save the adapter
    trainer.save_model("results/finetuned_model/final_adapter")
    print("Training complete and adapter saved.")

if __name__ == "__main__":
    train()
