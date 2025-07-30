import torch


class ModelConfig:
    DATASET_NAME = "sst2"
    MAX_SEQUENCE_LENGTH = 64
    NUM_SENTIMENT_CLASSES = 2

    # Model Params
    TRANSFORMER_MODEL_NAME = "distilbert-base-uncased"
    EMBEDDING_DIM = 768
    HIDDEN_DIM = 256
    NUM_HEADS = 4
    NUM_LAYERS = 2

    # Training Parameters
    BATCH_SIZE = 16
    LEARNING_RATE_G = 5e-5
    LEARNING_RATE_D = 5e-5
    NUM_EPOCHS = 10
    D_STEPS_PER_G_STEP = 1

    # Loss weights
    LAMBDA_ADV = 1.0  # weight for adversarial loss
    LAMBDA_SENTIMENT = 1.0  # weight for sentiment classification loss
    LAMBDA_CONTENT = (
        10.0  # weight for content preservation loss (higher to ensure meaning)
    )

    # Device Configuration
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

    if DEVICE == "mps":
        print("Using Apple Silicon MPS")
    else:
        print("using cpu")

    OUTPUT_DIR = "output/"
    GENERATED_SAMPLES_FILE = "generated_samples.txt"
    MODEL_SAVE_PATH = "models/"
