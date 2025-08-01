# Cross Modal Sentiment Style-Transfer with Adversarial Training

## Project Overview
This project explores a novel approach to natural language generation by implementing a Cross-Modal Sentiment and Style Transfer system using an Adversarial Training framework. The core idea is to take an input sentence and transform its sentiment (e.g., from negative to positive, or vice-versa) while preserving its original meaning and content, all powered by a Generator-Discriminator architecture.

While "cross-modal" often implies different data types (like text to image), in this context, it refers to the transfer of a specific linguistic attribute (sentiment/style) onto existing textual content, making it a "text-to-text" generation task with a controlled output characteristic.

This project is designed to showcase advanced deep learning techniques, including Transformer-based sequence generation, adversarial learning (inspired by GANs), and sophisticated loss functions for content preservation. It's built to be runnable on resource-constrained environments (like a laptop with 24GB RAM and limited SSD) by judiciously selecting model sizes and dataset subsets, while still delivering significant complexity and unique insights.

## Features
* Sentiment/Style Transfer: Transforms the sentiment of an input sentence to a target sentiment (e.g., negative to positive, positive to negative).

* Content Preservation: Employs a cosine similarity-based loss using pre-trained sentence embeddings to ensure the core meaning of the original sentence is maintained during transfer.

* Adversarial Training: Utilizes a Discriminator network to guide the Generator towards producing more realistic and sentiment-consistent outputs.

* Custom Transformer Architecture: The Generator is built using a custom, lightweight Transformer Encoder-Decoder structure, optimized for efficiency.

* Pre-trained Discriminator Backbone: The Discriminator leverages a DistilBERT model for robust feature extraction and sentiment classification.

* Containerized Environment: Fully packaged using Docker for easy setup and reproducibility across different environments.

* GPU Acceleration (MPS): Configured to leverage Apple's Metal Performance Shaders (MPS) for accelerated training on M-series Macs.

## Architecture
The project's core architecture consists of two primary deep learning models interacting in an adversarial manner, plus a helper model for content evaluation:

* Generator (G):

  - Type: Custom Transformer Encoder-Decoder.

  - Input: Original sentence token IDs and a target sentiment label.

  - Output: Logits for each token in the generated sentence sequence, aiming to match the target sentiment while retaining original content.

  - Role: The "creative" component, responsible for generating new text.

* Discriminator (D):

  - Type: DistilBERT (pre-trained) backbone with two linear classification heads.

  - Input: A sentence (either original or generated by G) token IDs.

  - Output 1 (Sentiment): Probability distribution over sentiment classes (e.g., negative/positive).

  - Output 2 (Real/Fake): Probability that the input sentence is "real" (from the original dataset) versus "fake" (generated by G).

  - Role: The "critic" component, guiding the Generator by providing feedback on sentiment correctness and generation realism.

* Content Preservation Model:

  - Type: Pre-trained SentenceTransformer (all-MiniLM-L6-v2).

  - Input: Raw text sentences (original and generated).

  - Output: High-dimensional semantic embeddings for sentences.

  - Role: Provides a stable metric for semantic similarity, ensuring the generated sentence doesn't lose the meaning of the original.

## Training Flow:

The training process alternates between updating the Discriminator and the Generator, driven by a multi-faceted loss function:

* Discriminator Loss: Combines:

  - Binary Cross-Entropy for distinguishing real vs. fake text.

  - Cross-Entropy for correctly classifying the sentiment of both real and generated text.

* Generator Loss: Combines:

  - Adversarial Loss: G tries to fool D into classifying its generated text as "real."

  - Sentiment/Style Loss: G tries to make D classify its generated text with the target sentiment.

  - Content Preservation Loss: G tries to maximize the cosine similarity between the embeddings of the original and generated sentences (i.e., minimize 1 - cosine_similarity).

## Dataset
This project utilizes the Stanford Sentiment Treebank v2 (SST-2) dataset.

 * Size: Approximately 67,349 training examples and 872 validation examples (for binary classification). This is a relatively small dataset, making it ideal for local development on machines with limited storage.

 * Content: Sentences from movie reviews labeled with binary sentiment (negative/positive).

 * Acquisition: Handled automatically by the Hugging Face datasets library within the project.

## Project Structure
```
sentiment_style_transfer/
├── app/
│   ├── __init__.py         # Python package marker
│   ├── main.py             # Main training and evaluation script
│   ├── models.py           # Defines Generator, Discriminator, and ContentPreservation models
│   ├── data_utils.py       # Handles data loading and preprocessing (SST-2)
│   └── config.py           # Stores hyper-parameters and configurations
├── Dockerfile              # Defines the Docker image for the project
├── requirements.txt        # Lists all Python dependencies
├── .dockerignore           # Specifies files/directories to exclude from the Docker build context
└── README.md               # This file
```
# Setup and Running
## Prerequisites

* Docker: Ensure Docker Desktop is installed and running on your system.

## Steps

+ Clone the Repository:
```bash
Bash
git clone https://github.com/your-username/sentiment_style_transfer.git
cd sentiment_style_transfer
```
(Replace https://github.com/your-username/sentiment_style_transfer.git with your actual repository URL).

+ Build the Docker Image:
This command builds the Docker image. It will download the necessary base image, install Python dependencies, and set up your application. This might take a few minutes on the first run.
```bash
Bash
docker build -t sentiment-style-transfer .
```
+ Run the Docker Container:
This command starts the Docker container, which will automatically execute the main.py training script.
```bash
Bash
docker run --rm --name sentiment_transfer_app sentiment-style-transfer
```
* ```--rm```: Automatically removes the container once it exits (useful for clean runs).

* ```--name sentiment_transfer_app```: Assigns a readable name to your container.

## Output

During training, you will see real-time progress updates in your terminal, including various loss values for the Generator and Discriminator.

* Generated Samples: After each epoch, the script will print a few example sentences, showing the original, target, and generated text, along with the sentiment predicted by the Discriminator for the generated text. These samples are also appended to output/generated_samples.txt.

* Model Checkpoints: Trained Generator and Discriminator models (PyTorch .pt files) will be saved in the models/ directory after each epoch.

## Configuration
All key parameters and hyperparameters can be found and modified in app/config.py. This includes:
```
* MAX_SEQUENCE_LENGTH

* BATCH_SIZE

* LEARNING_RATE_G, LEARNING_RATE_D

* NUM_EPOCHS

* LAMBDA_ADV, LAMBDA_SENTIMENT, LAMBDA_CONTENT (weights for loss components)

* TRANSFORMER_MODEL_NAME (for Discriminator backbone and Tokenizer)

* DEVICE (automatically set to mps for Apple Silicon if available)
```
