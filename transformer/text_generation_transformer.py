"""
Transformer Decoder for Text Generation
========================================
This example shows how to use a Transformer decoder (GPT-style) for
autoregressive text generation. Similar to GPT, this model is trained
to predict the next character/token given previous context.

Task: Learn to generate text character by character
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple


class CharacterDataset(Dataset):
    """
    Dataset for character-level language modeling.
    Creates sequences where the model predicts the next character.
    """

    def __init__(self, text: str, seq_len: int = 50):
        """
        Args:
            text: Input text corpus
            seq_len: Length of each training sequence
        """
        self.seq_len = seq_len

        # Create character vocabulary
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

        # Convert text to indices
        self.data = [self.char_to_idx[ch] for ch in text]

        # Calculate number of sequences
        self.num_sequences = len(self.data) - seq_len

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get sequence and target (shifted by 1)
        sequence = self.data[idx:idx + self.seq_len]
        target = self.data[idx + 1:idx + self.seq_len + 1]

        return (
            torch.LongTensor(sequence),
            torch.LongTensor(target)
        )


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    """
    Decoder-only Transformer (GPT-style) for text generation.
    Uses causal (autoregressive) masking to predict next token.
    """

    def __init__(
            self,
            vocab_size: int,
            d_model: int = 256,
            nhead: int = 8,
            num_layers: int = 6,
            dim_feedforward: int = 1024,
            dropout: float = 0.1,
            max_len: int = 512
    ):
        super().__init__()

        self.d_model = d_model

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask to prevent attending to future positions"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Input sequence (batch_size, seq_len)

        Returns:
            Logits (batch_size, seq_len, vocab_size)
        """
        # Embed and add positional encoding
        src_emb = self.pos_encoder(self.embedding(src) * np.sqrt(self.d_model))

        # Generate causal mask
        mask = self.generate_square_subsequent_mask(src.size(1)).to(src.device)

        # Since this is decoder-only, we use the same tensor for memory
        # This is the key difference from encoder-decoder architecture
        output = self.transformer_decoder(
            src_emb,
            src_emb,  # Use same input as memory
            tgt_mask=mask
        )

        # Project to vocabulary
        logits = self.fc_out(output)

        return logits

    @torch.no_grad()
    def generate(
            self,
            start_tokens: torch.Tensor,
            max_new_tokens: int,
            temperature: float = 1.0,
            top_k: int = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            start_tokens: Starting sequence (batch_size, seq_len)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k most likely tokens

        Returns:
            Generated sequence (batch_size, seq_len + max_new_tokens)
        """
        self.eval()

        generated = start_tokens

        for _ in range(max_new_tokens):
            # Get predictions for current sequence
            logits = self(generated)

            # Focus on last position
            logits = logits[:, -1, :] / temperature

            # Optionally apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample from distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

        return generated


def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(src)

        # Calculate loss
        loss = criterion(
            output.reshape(-1, output.size(-1)),
            tgt.reshape(-1)
        )

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device
) -> float:
    """Evaluate model"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src)
            loss = criterion(
                output.reshape(-1, output.size(-1)),
                tgt.reshape(-1)
            )
            total_loss += loss.item()

    return total_loss / len(dataloader)


def generate_text(
        model: TransformerDecoder,
        dataset: CharacterDataset,
        start_str: str,
        length: int = 100,
        temperature: float = 0.8,
        device: torch.device = torch.device('cpu')
) -> str:
    """
    Generate text starting from a given string.

    Args:
        model: Trained model
        dataset: Dataset (for character mappings)
        start_str: Starting text
        length: Number of characters to generate
        temperature: Sampling temperature
        device: Device to run on

    Returns:
        Generated text
    """
    model.eval()

    # Convert start string to indices
    start_indices = [dataset.char_to_idx[ch] for ch in start_str]
    input_seq = torch.LongTensor(start_indices).unsqueeze(0).to(device)

    # Generate
    with torch.no_grad():
        output_seq = model.generate(
            input_seq,
            max_new_tokens=length,
            temperature=temperature,
            top_k=10
        )

    # Convert back to text
    output_indices = output_seq[0].cpu().numpy()
    generated_text = ''.join([dataset.idx_to_char[idx] for idx in output_indices])

    return generated_text


def main():
    """Main training loop for character-level text generation"""

    # Sample text corpus (you would typically load a much larger corpus)
    sample_text = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them. To die—to sleep,
    No more; and by a sleep to say we end
    The heart-ache and the thousand natural shocks
    That flesh is heir to: 'tis a consummation
    Devoutly to be wish'd. To die, to sleep;
    To sleep, perchance to dream—ay, there's the rub:
    For in that sleep of death what dreams may come,
    When we have shuffled off this mortal coil,
    Must give us pause—there's the respect
    That makes calamity of so long life.
    """ * 10  # Repeat to have more training data

    # Hyperparameters
    SEQ_LEN = 50
    D_MODEL = 256
    NHEAD = 8
    NUM_LAYERS = 4
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.2
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001

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

    # Create dataset
    print("📚 Creating character-level dataset...")
    dataset = CharacterDataset(sample_text, seq_len=SEQ_LEN)
    print(f"   Vocabulary size: {dataset.vocab_size}")
    print(f"   Characters: {dataset.chars}")
    print(f"   Training sequences: {len(dataset)}\n")

    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Create model
    print("🏗️  Building Transformer decoder...")
    model = TransformerDecoder(
        vocab_size=dataset.vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_len=SEQ_LEN + 200
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {num_params:,}\n")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    print("🚀 Starting training...\n")
    best_val_loss = float('inf')
    current_lr = LEARNING_RATE

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        # Get current learning rate before scheduler step
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        # Print learning rate change if it happened
        lr_info = ""
        if new_lr != old_lr:
            lr_info = f" | LR: {old_lr:.6f} → {new_lr:.6f}"
            current_lr = new_lr

        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}{lr_info}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'text_generator.pt')

        # Generate sample text every 10 epochs
        if (epoch + 1) % 10 == 0:
            print("\n" + "="*70)
            print("SAMPLE GENERATION")
            print("="*70)
            sample = generate_text(
                model, dataset, "To be", length=150, temperature=0.8, device=device
            )
            print(sample)
            print("="*70 + "\n")

    print(f"\n✅ Training complete! Best validation loss: {best_val_loss:.4f}\n")

    # Final text generation examples
    print("\n" + "="*70)
    print("FINAL TEXT GENERATION EXAMPLES")
    print("="*70)

    prompts = ["To be", "And by", "The heart"]
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 70)
        generated = generate_text(
            model, dataset, prompt, length=200, temperature=0.7, device=device
        )
        print(generated)

    print("\n" + "="*70)
    print("💾 Model saved to 'text_generator.pt'")
    print("="*70)


if __name__ == "__main__":
    main()