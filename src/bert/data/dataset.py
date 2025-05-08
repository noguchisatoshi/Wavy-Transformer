from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer

def cast_label(examples: dict, data_type: str) -> dict:
    if data_type == "stsb":
        examples["label"] = [float(label) for label in examples["label"]]
    else:
        examples["label"] = [int(label) for label in examples["label"]]
    return examples

def get_text_columns(task_name: str) -> tuple:
    if task_name in ["mrpc", "rte", "wnli", "stsb",
                     "mnli", "mnli_matched", "mnli_mismatched"]:
        return ("premise" if task_name.startswith("mnli") else "sentence1",
                "hypothesis" if task_name.startswith("mnli") else "sentence2")
    elif task_name == "qqp":
        return ("question1", "question2")
    elif task_name == "qnli":
        return ("question", "sentence")
    elif task_name in ["cola", "sst2"]:
        return ("sentence", None)
    else:
        raise ValueError(f"Unsupported GLUE task: {task_name}")


def get_dataset(config: dict):
    data_type = config["data_type"]
    glue_task_list = config.get("glue_tasks", [])

    if data_type in glue_task_list:
        batch_size = config.get("batch_size", 32)
        max_length = config.get("max_length", 512)

        base_task = "mnli" if data_type.startswith("mnli") else data_type
        dataset = load_dataset("glue", base_task, cache_dir=config.get("cache_dir", None))

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        text_columns = get_text_columns(data_type)

        def preprocess_function(examples: dict) -> dict:
            if text_columns[1] is None:
                return tokenizer(
                    examples[text_columns[0]],
                    max_length=max_length,
                    truncation=True,
                    padding="max_length"
                )
            return tokenizer(
                examples[text_columns[0]],
                examples[text_columns[1]],
                max_length=max_length,
                truncation=True,
                padding="max_length"
            )

        dataset = dataset.map(preprocess_function, batched=True)

        train_data = dataset["train"]

        if data_type == "mnli_matched":
            val_split = "validation_matched"
        elif data_type == "mnli_mismatched":
            val_split = "validation_mismatched"
        elif data_type == "mnli":
            val_split = "validation_matched"
        else:
            val_split = "validation"

        val_data = dataset[val_split]

        train_data = train_data.map(lambda x: cast_label(x, data_type), batched=True)
        val_data   = val_data.map(lambda x: cast_label(x, data_type), batched=True)

        for ds in (train_data, val_data):
            ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)

    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    return train_loader, val_loader