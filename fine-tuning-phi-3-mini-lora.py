import logging
from random import randrange
import torch
from datasets import load_dataset, load_metric
from peft import LoraConfig, prepare_model_for_kbit_training, TaskType, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
    pipeline
)
from trl import SFTTrainer
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set seed for reproducibility
set_seed(42)

try:
    # Load dataset
    dataset_name = "mshojaei77/merged_persian_qa"
    dataset_split = "train"
    dataset = load_dataset(dataset_name, split=dataset_split)
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(dataset[randrange(len(dataset))])

    # Load model and tokenizer
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, add_eos_token=True, use_fast=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto",
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "sdpa"
    )
    logger.info("Model and tokenizer loaded successfully.")

    # Preprocess dataset
    def create_message_column(row):
        return {"messages": [
            {"role": "user", "content": f"متن زیر را بخوانید و به سوال پاسخ دهید:\n\n{row['Context']}"},
            {"role": "assistant", "content": f"{row['Response']}"}
        ]}

    def format_dataset_chatml(row):
        return {"text": tokenizer.apply_chat_template(row["messages"], tokenize=False)}

    dataset_chatml = dataset.map(create_message_column).map(format_dataset_chatml)
    dataset_chatml = dataset_chatml.train_test_split(test_size=0.05, seed=1234)
    logger.info("Dataset preprocessed successfully.")

    # Training arguments and PEFT config
    args = TrainingArguments(
        output_dir="./phi-3-mini-LoRA-persian-qa",
        evaluation_strategy="steps",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        warmup_ratio=0.1,
        logging_steps=100,
        eval_steps=100,
        report_to="wandb",
        fp16=True,
        seed=42,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"],
    )

    # Trainer setup
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_chatml['train'],
        eval_dataset=dataset_chatml['test'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=args,
    )
    logger.info("Trainer setup completed.")

    trainer.train()
    trainer.save_model()
    logger.info("Model training and saving completed.")

    # Evaluation
    rouge_metric = load_metric("rouge")

    def test_inference(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_length=512)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def calculate_rouge(row):
        response = test_inference(row['messages'][0]['content'])
        result = rouge_metric.compute(predictions=[response], references=[row['messages'][1]['content']], use_stemmer=True)
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        result['response'] = response
        return result

    metrics = dataset_chatml['test'].select(range(0, 500)).map(calculate_rouge, batched=False)
    logger.info("Evaluation completed.")

    logger.info("Rouge 1 Mean: %s", np.mean([m['rouge1'] for m in metrics]))
    logger.info("Rouge 2 Mean: %s", np.mean([m['rouge2'] for m in metrics]))
    logger.info("Rouge L Mean: %s", np.mean([m['rougeL'] for m in metrics]))
    logger.info("Rouge Lsum Mean: %s", np.mean([m['rougeLsum'] for m in metrics]))

    # Merge and save model
    new_model = PeftModel.from_pretrained(model, args.output_dir)
    merged_model = new_model.merge_and_unload()

    merged_model.save_pretrained("merged_model")
    tokenizer.save_pretrained("merged_model")
    logger.info("Model merged and saved successfully.")

    hf_model_repo = "your-username/your-model-name"  # Replace with your desired repo name

    merged_model.push_to_hub(hf_model_repo)
    tokenizer.push_to_hub(hf_model_repo)
    logger.info("Model and tokenizer pushed to Hugging Face Hub.")

except Exception as e:
    logger.error(f"An error occurred: {e}", exc_info=True)
