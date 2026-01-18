"""
Transformer Training Example in PyTorch
========================================
This example trains a Transformer model from scratch on a sequence reversal task.
The model learns to reverse input sequences, demonstrating core Transformer concepts.

Task: Given input sequence [1, 2, 3, 4, 5], predict [5, 4, 3, 2, 1]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


class SequenceReversalDataset(Dataset):
    """
    Dataset that generates random sequences and their reversals.
    This is a simple task that requires the model to learn attention patterns.
    """

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        """
        Args:
            num_samples: Number of sequences to generate
            seq_len: Length of each sequence
            vocab_size: Size of vocabulary (number of unique tokens)
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Generate random sequences (excluding 0 which we'll use for padding)
        self.sequences = np.random.randint(1, vocab_size, size=(num_samples, seq_len))
        # Reverse sequences for targets
        self.targets = np.flip(self.sequences, axis=1).copy()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.LongTensor(self.sequences[idx]),
            torch.LongTensor(self.targets[idx])
        )


class PositionalEncoding(nn.Module):
    """
    Adds positional information to embeddings using sine/cosine functions.
    This allows the model to understand token positions in sequences.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence tasks.
    Uses PyTorch's built-in Transformer layers.
    """

    def __init__(
            self,
            vocab_size: int,
            d_model: int = 128,
            nhead: int = 8,
            num_encoder_layers: int = 3,
            num_decoder_layers: int = 3,
            dim_feedforward: int = 512,
            dropout: float = 0.1,
            max_len: int = 100
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Dimension of embeddings and model
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super().__init__()

        self.d_model = d_model

        # Embedding layers
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Use (batch, seq, feature) format
        )

        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through the Transformer.

        Args:
            src: Source sequence (batch_size, src_len)
            tgt: Target sequence (batch_size, tgt_len)
            tgt_mask: Mask for target sequence to prevent looking ahead

        Returns:
            Logits of shape (batch_size, tgt_len, vocab_size)
        """
        # Embed and add positional encoding
        src_emb = self.pos_encoder(self.encoder_embedding(src) * np.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.decoder_embedding(tgt) * np.sqrt(self.d_model))

        # Pass through transformer
        transformer_out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask
        )

        # Project to vocabulary
        output = self.fc_out(transformer_out)

        return output

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate a causal mask to prevent attention to future positions.
        This is crucial for autoregressive generation.

        Args:
            sz: Size of the square mask

        Returns:
            Mask tensor of shape (sz, sz)
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device
) -> float:
    """
    Train for one epoch.

    Args:
        model: The Transformer model
        dataloader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)

        # Create target input (shifted right by prepending a start token)
        # For simplicity, we use the first token of target as start token
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Generate causal mask
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(src, tgt_input, tgt_mask)

        # Calculate loss
        loss = criterion(
            output.reshape(-1, output.size(-1)),
            tgt_output.reshape(-1)
        )

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate model on validation set.

    Args:
        model: The Transformer model
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)

            output = model(src, tgt_input, tgt_mask)

            loss = criterion(
                output.reshape(-1, output.size(-1)),
                tgt_output.reshape(-1)
            )

            total_loss += loss.item()

            # Calculate accuracy
            predictions = output.argmax(dim=-1)
            correct += (predictions == tgt_output).sum().item()
            total += tgt_output.numel()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def visualize_training(train_losses: List[float], val_losses: List[float], val_accuracies: List[float]):
    """Create visualization of training progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, val_accuracies, 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    # Save to current directory (wherever you run the script from)
    plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    print("📊 Training visualization saved to 'training_progress.png'")


def test_model(model: nn.Module, dataset: Dataset, device: torch.device, num_examples: int = 5):
    """
    Test the model on a few examples and display results.
    """
    model.eval()
    print("\n" + "="*70)
    print("MODEL PREDICTIONS")
    print("="*70)

    with torch.no_grad():
        for i in range(num_examples):
            src, tgt = dataset[i]
            src = src.unsqueeze(0).to(device)
            tgt = tgt.unsqueeze(0).to(device)

            # Start with first token
            decoder_input = tgt[:, :1]

            # Autoregressively generate sequence
            for _ in range(tgt.size(1) - 1):
                tgt_mask = model.generate_square_subsequent_mask(decoder_input.size(1)).to(device)
                output = model(src, decoder_input, tgt_mask)
                next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                decoder_input = torch.cat([decoder_input, next_token], dim=1)

            src_seq = src.cpu().numpy()[0]
            tgt_seq = tgt.cpu().numpy()[0]
            pred_seq = decoder_input.cpu().numpy()[0, 1:]  # Skip start token

            match = "✓" if np.array_equal(pred_seq, tgt_seq[1:]) else "✗"

            print(f"\nExample {i+1} {match}")
            print(f"  Input:    {src_seq}")
            print(f"  Target:   {tgt_seq}")
            print(f"  Predicted: [0, {', '.join(map(str, pred_seq))}]")


def main():
    """Main training loop"""

    # Hyperparameters
    VOCAB_SIZE = 20 # The number of unique tokens (words, characters, or numbers) in your vocabulary.
    SEQ_LEN = 10 # Maximum sequence length the model can process at once. (Defines the context window)
    D_MODEL = 128 # The dimensionality of the model's internal representations (embedding size). All vectors inside the Transformer have this dimension
    NHEAD = 8 # Number of parallel attention heads in multi-head attention.
    NUM_ENCODER_LAYERS = 3 # Number of stacked encoder layers.
    NUM_DECODER_LAYERS = 3 # Number of stacked decoder layers.
    DIM_FEEDFORWARD = 512 # Hidden dimension of the position-wise feedforward network inside each Transformer layer.How it works: After attention, each position goes through a 2-layer MLP:
    DROPOUT = 0.1 # Probability of randomly "dropping out" (setting to zero) neurons during training.
    BATCH_SIZE = 64 # Number of examples processed together in one forward/backward pass.
    NUM_EPOCHS = 50 # Number of complete passes through the entire training dataset.
    LEARNING_RATE = 0.0001 # Step size for weight updates during gradient descent.

    # Device - optimized for Apple Silicon (MPS), CUDA, or CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"🖥️  Using device: MPS (Apple Silicon GPU)\n")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🖥️  Using device: CUDA (NVIDIA GPU)\n")
    else:
        device = torch.device('cpu')
        print(f"🖥️  Using device: CPU\n")

    # Create datasets
    print("📊 Creating datasets...")
    train_dataset = SequenceReversalDataset(5000, SEQ_LEN, VOCAB_SIZE)
    val_dataset = SequenceReversalDataset(1000, SEQ_LEN, VOCAB_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Create model
    print("🏗️  Building model...")
    model = TransformerModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_len=SEQ_LEN + 10
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total trainable parameters: {num_params:,}\n")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("🚀 Starting training...\n")
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        # Evaluate
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Print progress
        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_transformer_model.pt')

    print(f"\n✅ Training complete! Best validation loss: {best_val_loss:.4f}\n")

    # Visualize training
    visualize_training(train_losses, val_losses, val_accuracies)

    # Test model
    test_model(model, val_dataset, device)

    print("\n" + "="*70)
    print("📁 Model saved to 'best_transformer_model.pt'")
    print("="*70)


if __name__ == "__main__":
    main()