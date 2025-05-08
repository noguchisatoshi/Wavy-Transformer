import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import yaml
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertConfig, BertForQuestionAnswering, AdamW, get_scheduler
from datasets import load_dataset
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.bert import CustomBertForQuestionAnswering

def prepare_features(examples, tokenizer, max_length=512, doc_stride=128):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    
    start_positions = []
    end_positions = []
    for i, offsets in enumerate(offset_mapping):
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        if len(answers["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            sequence_ids = tokenized_examples.sequence_ids(i)
            token_start_index = 0
            while token_start_index < len(sequence_ids) and sequence_ids[token_start_index] != 1:
                token_start_index += 1
            token_end_index = len(offsets) - 1
            while token_end_index >= 0 and sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                start_positions.append(0)
                end_positions.append(0)
            else:
                for idx in range(token_start_index, token_end_index + 1):
                    if offsets[idx][0] <= start_char < offsets[idx][1]:
                        start_positions.append(idx)
                        break
                for idx in range(token_end_index, token_start_index - 1, -1):
                    if offsets[idx][1] >= end_char:
                        end_positions.append(idx)
                        break
    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="finetune_config.yaml",
                        help="finetune用の設定ファイル(YAML)")
    parser.add_argument("--finetune_bert_base", action="store_true",
                        help="Trueの場合、Hugging Face の BertForQuestionAnswering を利用してファインチューニングする")
    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    
    model_config = config_dict["model"]
    residual_type = model_config.get("residual_type", "diffuse")
    tau = model_config.get("tau", 1.0)
    add_mass = model_config.get("add_mass", False)
    pretrain_model_path = model_config.get("pretrain_model_path", None)
    
    training_config = config_dict["training"]
    max_seq_length = training_config.get("max_seq_length", 512)
    per_device_batch_size = training_config.get("per_device_batch_size", 32)
    gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 2)
    num_epochs = training_config.get("num_epochs", 2)
    learning_rate = training_config.get("learning_rate", 0.00005)
    max_steps = training_config.get("max_steps", 100000)
    warmup_proportion = training_config.get("warmup_proportion", 0.1)
    weight_decay_list = training_config.get("weight_decay", [0.1, 0.01])
    lr_scheduler_type = training_config.get("lr_scheduler", "linear")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    vocab_size = model_config.get("vocab_size", 30522)
    hidden_size = model_config.get("hidden_size", 768)
    intermediate_size = model_config.get("intermediate_size", 3072)
    num_hidden_layers = model_config.get("num_hidden_layers", 12)
    num_attention_heads = model_config.get("num_attention_heads", 12)
    max_position_embeddings = model_config.get("max_position_embeddings", 512)
    hidden_dropout_prob = model_config.get("dropout_prob", 0.1)
    
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        max_position_embeddings=max_position_embeddings,
        hidden_dropout_prob=hidden_dropout_prob,
    )
    
    if args.finetune_bert_base:
        model = BertForQuestionAnswering.from_pretrained("bert-base-uncased", config=config)
        print("Using Hugging Face's BertForQuestionAnswering for fine-tuning.")
    else:
        model = CustomBertForQuestionAnswering(
            config,
            residual_type=residual_type,
            tau=tau
        )
        print("Using custom BERT model for fine-tuning.")
    model.to(device)
    
    if pretrain_model_path is not None:
        state_dict = torch.load(pretrain_model_path, map_location=device)
        if args.finetune_bert_base:
            missing_keys, unexpected_keys = model.bert.load_state_dict(state_dict, strict=False)
            print("Pretrained BERT state loaded. Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
        else:
            missing_keys, unexpected_keys = model.bert.load_state_dict(state_dict, strict=False)
            print("Pretrain model loaded into custom BERT. Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
    
    squad_dataset = load_dataset("squad_v2")
    
    tokenized_train = squad_dataset["train"].map(
        lambda examples: prepare_features(examples, tokenizer, max_length=max_seq_length, doc_stride=max_seq_length//4),
        batched=True,
        remove_columns=squad_dataset["train"].column_names
    )
    tokenized_val = squad_dataset["validation"].map(
        lambda examples: prepare_features(examples, tokenizer, max_length=max_seq_length, doc_stride=max_seq_length//4),
        batched=True,
        remove_columns=squad_dataset["validation"].column_names
    )
    tokenized_train.set_format(type="torch")
    tokenized_val.set_format(type="torch")
    
    train_loader = DataLoader(tokenized_train, batch_size=per_device_batch_size, shuffle=True)
    dev_loader = DataLoader(tokenized_val, batch_size=per_device_batch_size)
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay_list[0],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay_list[1],
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    
    total_training_steps = (len(train_loader) // gradient_accumulation_steps) * num_epochs
    warmup_steps = int(warmup_proportion * total_training_steps)

    scheduler = get_scheduler(
        lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    
    global_step = 0
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)
            
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                start_positions=start_positions,
                end_positions=end_positions
            )
            loss = outputs[0]
            loss = loss / gradient_accumulation_steps
            loss.backward()
            epoch_loss += loss.item()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                progress_bar.set_postfix({
                    "global_step": global_step,
                    "loss": loss.item()
                })
                
                if global_step >= max_steps:
                    break
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            eval_bar = tqdm(dev_loader, desc="Evaluating", leave=False)
            for batch in eval_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)
                
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    start_positions=start_positions,
                    end_positions=end_positions
                )
                eval_loss += outputs[0].item()
                eval_bar.set_postfix({
                    "eval_loss": outputs[0].item()
                })
        avg_eval_loss = eval_loss / len(dev_loader)
        model.train()
        
        print(f"Epoch {epoch+1}/{num_epochs} - Global Step: {global_step} - Train Loss: {avg_train_loss:.4f} - Eval Loss: {avg_eval_loss:.4f}")
        
        if global_step >= max_steps:
            print("Training stopped")
            break


    final_model_path = f"/workspace/nas/oversmooth_bert/src/finetune_SQuAD/finetune_models/finetune_model_{residual_type}.pt"
    if args.finetune_bert_base:
        torch.save(model.bert.state_dict(), final_model_path)
    else:
        torch.save(model.bert.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()