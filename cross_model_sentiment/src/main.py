import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sentence_transformers import util
from tqdm import tqdm
import os

from modelconfig import ModelConfig
from data_utils import get_dataloaders
from models import Generator, Discriminator, ContentPreservationModel

os.makedirs(ModelConfig.OUTPUT_DIR, exist_ok=True)
os.makedirs(ModelConfig.MODEL_SAVE_PATH, exist_ok=True)


def train():
    device = torch.device(ModelConfig.DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.TRANSFORMER_MODEL_NAME)
    vocab_size = tokenizer.vocab_size

    train_loader, val_loader = get_dataloaders(tokenizer, ModelConfig.BATCH_SIZE)

    generator = Generator(
        vocab_size=vocab_size,
        embedding_dim=ModelConfig.EMBEDDING_DIM,
        hidden_dim=ModelConfig.HIDDEN_DIM,
        num_heads=ModelConfig.NUM_HEADS,
        num_layers=ModelConfig.NUM_LAYERS,
        max_seq_len=ModelConfig.MAX_SEQUENCE_LENGTH,
        num_sentiment_classes=ModelConfig.NUM_SENTIMENT_CLASSES,
    ).to(device)

    discriminator = Discriminator(
        transformer_model_name=ModelConfig.TRANSFORMER_MODEL_NAME,
        num_sentiment_classes=ModelConfig.NUM_SENTIMENT_CLASSES,
    ).to(device)

    content_preserver = ContentPreservationModel().to(device)

    optimizer_G = optim.AdamW(generator.parameters(), lr=ModelConfig.LEARNING_RATE_G)
    optimizer_D = optim.AdamW(
        discriminator.parameters(), lr=ModelConfig.LEARNING_RATE_D
    )

    bce_loss = nn.BCEWithLogitsLoss()  # For adversarial real / fake classification
    ce_loss = nn.CrossEntropyLoss()  # for sentiment classification

    real_labels = torch.ones(ModelConfig.BATCH_SIZE, 1).to(device)
    fake_labels = torch.ones(ModelConfig.BATCH_SIZE, 1).to(device)

    for epoch in range(ModelConfig.NUM_EPOCHS):
        generator.train()
        discriminator.train()
        total_g_loss = 0
        total_d_loss = 0

        progress_bar = tqdm(
            enumerate(train_loader), total=len(train_loader), desc=f"epoch : {epoch+1}"
        )
        for i, batch in progress_bar:
            original_input_ids = batch["original_input_ids"].to(device)
            original_attention_mask = batch["original_attention_mask"].to(device)
            original_label = batch["original_label"].to(device)
            target_label = batch["target_label"].to(device)
            original_sentence_str = batch["original_sentence_str"]

            current_batch_size = original_input_ids.size(0)

            current_real_labels = torch.ones(current_batch_size, 1).to(device)
            current_fake_labels = torch.ones(current_batch_size, 1).to(device)

            # Training the Discriminator
            optimizer_D.zero_grad()

            real_sentiment_logits, real_fake_logits = discriminator(
                original_input_ids, original_attention_mask
            )
            d_real_loss = bce_loss(real_fake_logits, current_real_labels)
            d_real_sentiment_loss = ce_loss(real_sentiment_logits, original_label)

            d_real_total_loss = d_real_loss + d_real_sentiment_loss

            with torch.no_grad():
                generated_logits = generator(
                    original_input_ids, original_attention_mask, target_label
                )
                generated_token_ids = torch.argmax(generated_logits, dim=1)

                generated_sentences_str = tokenizer.decode(
                    generated_token_ids, skip_special_tokens=True
                )

            generated_encoding = tokenizer(
                generated_sentences_str,
                max_length=ModelConfig.MAX_SEQUENCE_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)

            fake_sentiment_logits, fake_real_fake_logits = discriminator(
                generated_encoding["input_ids"], generated_encoding["attention_mask"]
            )

            d_fake_loss = bce_loss(fake_real_fake_logits, current_fake_labels)
            d_fake_sentiment_loss = ce_loss(fake_sentiment_logits, target_label)

            d_loss = d_real_total_loss + d_fake_loss + d_fake_sentiment_loss
            d_loss.backward()
            optimizer_D.step()
            total_d_loss += d_loss.item()

            # Training the Generator
            for _ in range(ModelConfig.D_STEPS_PER_G_STEP):
                optimizer_G.zero_grad()

                generated_logits = generator(
                    original_input_ids, original_attention_mask, target_label
                )
                generated_token_ids = torch.argmax(generated_logits, dim=-1)
                generated_sentences_str = tokenizer.batch_decode(
                    generated_token_ids, skip_special_tokens=True
                )

                generated_encoding_for_g = tokenizer(
                    generated_sentences_str,
                    max_length=ModelConfig.MAX_SEQUENCE_LENGTH,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(device)

                g_sentiment_logits, g_real_fake_logits = discriminator(
                    generated_encoding_for_g["input_ids"],
                    generated_encoding_for_g["attention_mask"],
                )

                g_adv_loss = bce_loss(
                    g_real_fake_logits, current_real_labels
                )  # G wants D to output 1 (real)
                g_sentiment_loss = ce_loss(g_sentiment_logits, target_label)

                original_embeddings = content_preserver(original_sentence_str)
                generated_embeddings = content_preserver(generated_sentences_str)

                content_similarity = util.cos_sim(
                    original_embeddings, generated_embeddings
                )

                content_loss = (1 - content_similarity.diag()).mean()

                g_loss = (
                    ModelConfig.LAMBDA_ADV * g_adv_loss
                    + ModelConfig.LAMBDA_SENTIMENT * g_sentiment_loss
                    + ModelConfig.LAMBDA_CONTENT * content_loss
                )

                g_loss.backward()
                optimizer_G.step()
                total_g_loss += g_loss.item()

            progress_bar.set_postfix(
                d_loss=f"{d_loss.item():.4f}",
                g_loss=f"{g_loss.item():.4f}",
                g_adv=f"{g_adv_loss.item():.4f}",
                g_sent=f"{g_sentiment_loss.item():.4f}",
                g_cont=f"{content_loss.item():.4f}",
            )
    print(
        f"Epoch {epoch+1} finished. Avg D Loss: {total_d_loss / len(train_loader):.4f}, Avg G Loss: {total_g_loss / len(train_loader):.4f}"
    )

    # Evaluation
    generator.eval()
    discriminator.eval()

    num_samples_to_generate = 5
    generated_samples = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples_to_generate:
                break
            original_input_ids = batch["original_input_ids"].to(device)
            original_attention_mask = batch["original_attention_mask"].to(device)
            original_label = batch["original_label"].to(device)
            target_label = batch["target_label"].to(device)
            original_sentence_str = batch["original_sentence_str"]

            generated_logits = generator(
                original_input_ids, original_attention_mask, target_label
            )
            generated_token_ids = torch.argmax(
                generated_logits, dim=-1
            )  # Greedy decoding

            for j in range(len(original_sentence_str)):
                original_sent = original_sentence_str[j]
                gen_sent_tokens = generated_token_ids[j].cpu().tolist()
                generated_sent = tokenizer.decode(
                    gen_sent_tokens, skip_special_tokens=True
                )

                # Get sentiment prediction of generated sentence from D
                gen_encoding_for_disc = tokenizer(
                    generated_sent,
                    max_length=ModelConfig.MAX_SEQUENCE_LENGTH,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(device)

                disc_gen_sentiment_logits, _ = discriminator(
                    gen_encoding_for_disc["input_ids"],
                    gen_encoding_for_disc["attention_mask"],
                )
                predicted_gen_sentiment = torch.argmax(
                    disc_gen_sentiment_logits, dim=-1
                ).item()

                label_map = {0: "negative", 1: "positive"}

                sample_info = (
                    f"Original ({label_map[original_label[j].item()]}): {original_sent}\n"
                    f"Target ({label_map[target_label[j].item()]}):\n"
                    f"Generated (Predicted by D as {label_map[predicted_gen_sentiment]}): {generated_sent}\n"
                    f"{'-'*50}"
                )
                generated_samples.append(sample_info)
                print(sample_info)

    with open(
        os.path.join(ModelConfig.OUTPUT_DIR, ModelConfig.GENERATED_SAMPLES_FILE), "a"
    ) as f:
        f.write(f"Epoch {epoch+1} Samples\n")
        for sample in generated_samples:
            f.write(sample + "\n")

    torch.save(
        generator.state_dict(),
        os.path.join(ModelConfig.MODEL_SAVE_PATH, f"generator_epoch_{epoch+1}.pt"),
    )
    torch.save(
        discriminator.state_dict(),
        os.path.join(ModelConfig.MODEL_SAVE_PATH, f"discriminator_epoch_{epoch+1}.pt"),
    )
    print(f"Models saved for epoch {epoch+1}")


if __name__ == "__main__":
    train()
