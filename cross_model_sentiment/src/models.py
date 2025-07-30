import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from modelconfig import ModelConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0) / d_model))
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class Generator(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_heads,
        num_layers,
        max_seq_len,
        num_sentiment_classes,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.sentiment_embedding = nn.Embedding(num_sentiment_classes, embedding_dim)
        self.decoder_linear = nn.Linear(embedding_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder_linear.bias.data.zero_()
        self.decoder_linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_ids, attention_mask, target_sentiment_label):
        src = self.token_embedding(input_ids) * torch.sqrt(
            torch.tensor(src.size(-1), dtype=torch.float32)
        )
        src = self.positional_encoding(src)

        src_key_padding_mask = attention_mask == 0

        memory = self.transformer_encoder(
            src, src_key_padding_mask=src_key_padding_mask
        )

        sentiment_emb = self.sentiment_embedding(target_sentiment_label).unsqueeze(1)
        sentiment_emb = sentiment_emb.expand(-1, memory.size(1), -1)
        combined_memory = memory * sentiment_emb

        logits = self.decoder_linear(combined_memory)

        return logits


class Discriminator(nn.Module):
    def __init__(self, transformer_model_name, num_sentiment_classes):
        super().__init__()
        """
        Using a pre-trained model for robust feature extraction
        """
        self.encoder = AutoModel.from_pretrained(transformer_model_name)

        self.dropout = nn.Dropout(0.1)

        self.sentiment_classifier = nn.Linear(
            self.encoder.config.hidden_size, num_sentiment_classes
        )
        self.real_fake_classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """
        Get the [CLS] token embedding as the sequence representation
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        pooled_output = self.dropout(pooled_output)

        sentiment_logits = self.sentiment_classifier
        real_fake_logits = self.real_fake_classifier

        return sentiment_logits, real_fake_logits


"""
Sentence Transformer for sentence embedding. 
I use SentenceTransformer, it's pre trained and provides robust sentence embeddings
"""
from sentence_transformers import SentenceTransformer


class ContentPreservationModel(nn.Module):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super().__init__()

        self.model = SentenceTransformer(model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, sentences):
        embeddings = self.model.encode(
            sentences, convert_to_tensor=True, device=ModelConfig.DEVICE
        )
        return embeddings


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.TRANSFORMER_MODEL_NAME)
    vocab_size = tokenizer.vocab_size

    gen_model = Generator(
        vocab_size=vocab_size,
        embedding_dim=ModelConfig.EMBEDDING_DIM,
        hidden_dim=ModelConfig.HIDDEN_DIM,
        num_heads=ModelConfig.NUM_HEADS,
        num_layers=ModelConfig.NUM_LAYERS,
        max_seq_len=ModelConfig.MAX_SEQUENCE_LENGTH,
        num_sentiment_classes=ModelConfig.NUM_SENTIMENT_CLASSES,
    ).to(ModelConfig.DEVICE)

    dummy_input_ids = torch.randint(
        0, vocab_size, (ModelConfig.BATCH_SIZE, ModelConfig.MAX_SEQUENCE_LENGTH)
    ).to(ModelConfig.DEVICE)
    dummy_attention_mask = torch.ones(
        ModelConfig.BATCH_SIZE, ModelConfig.MAX_SEQUENCE_LENGTH
    ).to(ModelConfig.DEVICE)
    dummy_target_sentiment = torch.randint(
        0, ModelConfig.NUM_SENTIMENT_CLASSES, (ModelConfig.BATCH_SIZE,)
    ).to(ModelConfig.DEVICE)

    gen_logits = gen_model(
        dummy_input_ids, dummy_attention_mask, dummy_target_sentiment
    )

    disc_model = Discriminator(
        transformer_model_name=ModelConfig.TRANSFORMER_MODEL_NAME,
        num_sentiment_classes=ModelConfig.NUM_SENTIMENT_CLASSES,
    ).to(ModelConfig.DEVICE)

    disc_sentiment_logits, disc_real_fake_logits = disc_model(
        dummy_input_ids, dummy_attention_mask
    )

    content_model = ContentPreservationModel().to(ModelConfig.DEVICE)
    dummy_sentences = [
        "This is a test sentence.",
        "Another test sentence for embeddings.",
    ]
    embeddings = content_model(dummy_sentences)
