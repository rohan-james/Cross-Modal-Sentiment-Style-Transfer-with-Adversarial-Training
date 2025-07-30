from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from modelconfig import ModelConfig


class SentimentTransferDataset(Dataset):
    def __init__(self, tokenizer, data_split="train"):
        self.dataset = load_dataset(ModelConfig.DATASET_NAME)
        self.tokenizer = tokenizer
        self.max_seq_len = ModelConfig.MAX_SEQUENCE_LENGTH

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        original_sentence = str(item["sentence"])
        original_label = int(item["label"])

        # Adversarial training
        target_label = 1 - original_label

        original_encoding = self.tokenizer(
            original_sentence,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "original_input_ids": original_encoding["input_ids"].squeeze(),
            "original_attention_mask": original_encoding["attention_mask"].squeeze(),
            "original_label": torch.tensor(original_label, dtype=torch.long),
            "target_label": torch.tensor(target_label, dtype=torch.long),
            "original_sentence_str": original_sentence,
        }


def get_dataloaders(tokenizer, batch_size=ModelConfig.BATCH_SIZE):
    train_dataset = SentimentTransferDataset(tokenizer, data_split="train")
    validation_dataset = SentimentTransferDataset(tokenizer, data_split="validation")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.TRANSFORMER_MODEL_NAME)
    train_loader, val_loader = get_dataloaders(tokenizer)
