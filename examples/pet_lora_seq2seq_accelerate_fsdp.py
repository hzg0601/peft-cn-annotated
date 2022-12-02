import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup

from datasets import load_dataset
from pet import get_pet_config, get_pet_model, get_pet_model_state_dict
from pet.utils.other import fsdp_auto_wrap_policy
from tqdm import tqdm


def main():
    accelerator = Accelerator()
    model_name_or_path = "bigscience/mt0-xxl"
    batch_size = 16
    text_column = "sentence"
    label_column = "text_label"
    max_length = 64
    lr = 1e-3
    num_epochs = 1

    config = {"pet_type": "LORA", "task_type": "SEQ_2_SEQ_LM", "r": 8, "lora_alpha": 32, "lora_dropout": 0.1}
    pet_config = get_pet_config(config)
    checkpoint_name = "financial_sentiment_analysis_lora_fsdp_v1.pt"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model = get_pet_model(model, pet_config)
    accelerator.print(model.print_trainable_parameters())

    dataset = load_dataset("financial_phrasebank", "sentences_allagree")
    dataset = dataset["train"].train_test_split(test_size=0.1)
    dataset["validation"] = dataset["test"]
    del dataset["test"]

    classes = dataset["train"].features["label"].names
    dataset = dataset.map(
        lambda x: {"text_label": [classes[label] for label in x["label"]]},
        batched=True,
        num_proc=1,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[label_column]
        model_inputs = tokenizer(
            inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        labels = tokenizer(targets, max_length=3, padding="max_length", truncation=True, return_tensors="pt")
        labels = labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs

    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    if accelerator.state.fsdp_plugin is not None:
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)

    model, train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, eval_dataloader, optimizer, lr_scheduler
    )
    accelerator.print(model)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(
                    accelerator.gather_for_metrics(torch.argmax(outputs.logits, -1)).detach().cpu().numpy(),
                    skip_special_tokens=True,
                )
            )

        eval_epoch_loss = eval_loss / len(train_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(eval_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

        correct = 0
        total = 0
        for pred, true in zip(eval_preds, dataset["validation"][label_column]):
            if pred.strip() == true.strip():
                correct += 1
            total += 1
        accuracy = correct / total * 100
        accelerator.print(f"{accuracy=}")
        accelerator.print(f"{eval_preds[:10]=}")
        accelerator.wait_for_everyone()
        accelerator.save(get_pet_model_state_dict(model), checkpoint_name)
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
