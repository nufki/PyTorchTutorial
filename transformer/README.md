# PyTorch Transformer Training Example

This is a complete, from-scratch implementation of training a Transformer model in PyTorch. The example uses a sequence reversal task to demonstrate core concepts.

## 📋 Table of Contents
- [Overview](#overview)
- [Key Components](#key-components)
- [How It Works](#how-it-works)
- [Running the Code](#running-the-code)
- [Understanding the Output](#understanding-the-output)
- [Extending to Other Tasks](#extending-to-other-tasks)
- [Advanced Concepts](#advanced-concepts)

## 🎯 Overview

The model learns to reverse sequences:
- **Input**: `[1, 2, 3, 4, 5]`
- **Output**: `[5, 4, 3, 2, 1]`

This simple task requires the model to:
1. Attend to all input positions
2. Learn positional relationships
3. Generate outputs autoregressively

## 🔧 Key Components

### 1. **Positional Encoding**
```python
class PositionalEncoding(nn.Module):
    # Adds position information using sine/cosine functions
    # Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    #          PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Why it matters**: Transformers have no inherent notion of sequence order. Positional encoding injects this information.

### 2. **Causal Mask**
```python
def generate_square_subsequent_mask(self, sz: int):
    # Creates upper triangular matrix of -inf values
    # Prevents decoder from attending to future positions
```

**Example mask** (3x3):
```
[[0,   -inf, -inf],
 [0,    0,   -inf],
 [0,    0,    0  ]]
```

This ensures position `i` can only attend to positions `≤ i`.

### 3. **Embedding Scaling**
```python
src_emb = self.encoder_embedding(src) * np.sqrt(self.d_model)
```

**Why**: Scaling by √d_model prevents positional encodings from dominating the embeddings.

### 4. **Teacher Forcing**
```python
tgt_input = tgt[:, :-1]   # Use ground truth as input
tgt_output = tgt[:, 1:]    # Predict next token
```

During training, we feed the correct previous tokens rather than model predictions. This speeds up training significantly.

## ⚙️ How It Works

### Forward Pass

```
Input Sequence: [1, 2, 3, 4, 5]
       ↓
1. Token Embedding (vocab_size → d_model)
   [1, 2, 3, 4, 5] → [[0.1, 0.3, ...], [0.2, 0.4, ...], ...]
       ↓
2. Positional Encoding (add position info)
   + [[sin(0), cos(0), ...], [sin(1), cos(1), ...], ...]
       ↓
3. Encoder (self-attention + feedforward)
   - Multi-head attention: each token attends to all tokens
   - Feedforward: process each position independently
       ↓
4. Decoder (masked self-attention + cross-attention + feedforward)
   - Masked self-attention: attend to previous output tokens only
   - Cross-attention: attend to encoder output
   - Feedforward: final processing
       ↓
5. Output Projection (d_model → vocab_size)
   → [[0.9 for '5', 0.05 for '4', ...], ...]
       ↓
6. Argmax
   → [5, 4, 3, 2, 1]
```

### Training Loop

```python
for epoch in epochs:
    for batch in dataloader:
        # 1. Forward pass
        output = model(src, tgt_input, tgt_mask)
        
        # 2. Calculate loss (cross-entropy)
        loss = criterion(output, tgt_output)
        
        # 3. Backward pass
        loss.backward()
        
        # 4. Gradient clipping (prevents exploding gradients)
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 5. Update weights
        optimizer.step()
```

## 🚀 Running the Code

### Installation
```bash
pip install torch numpy matplotlib
```

### Run Training
```bash
python transformer_training.py
```

### Expected Output
```
🖥️  Using device: cuda
📊 Creating datasets...
🏗️  Building model...
   Total trainable parameters: 1,234,567
🚀 Starting training...

Epoch   1/50 | Train Loss: 2.8456 | Val Loss: 2.7123 | Val Acc: 0.1234
Epoch   2/50 | Train Loss: 2.3456 | Val Loss: 2.2123 | Val Acc: 0.2345
...
Epoch  50/50 | Train Loss: 0.0123 | Val Loss: 0.0234 | Val Acc: 0.9987

✅ Training complete! Best validation loss: 0.0234

======================================================================
MODEL PREDICTIONS
======================================================================

Example 1 ✓
  Input:    [ 5 12  3  8 15  2  9  6 11  4]
  Target:   [ 4 11  6  9  2 15  8  3 12  5]
  Predicted: [ 4 11  6  9  2 15  8  3 12  5]
```

## 📊 Understanding the Output

### Training Curves
The script generates `training_progress.png` showing:
1. **Loss curves**: Should decrease over time
2. **Accuracy**: Should increase, ideally reaching >99% for this simple task

### Model Predictions
The test examples show:
- ✓ means perfect prediction
- ✗ means at least one error

For sequence reversal, the model should achieve near-perfect accuracy after sufficient training.

## 🎓 Extending to Other Tasks

### 1. **Translation Task**
```python
# Modify dataset to return (english, spanish) pairs
class TranslationDataset(Dataset):
    def __init__(self, pairs):
        self.src_sentences = [pair[0] for pair in pairs]
        self.tgt_sentences = [pair[1] for pair in pairs]
    
    def __getitem__(self, idx):
        return (
            tokenize(self.src_sentences[idx]),
            tokenize(self.tgt_sentences[idx])
        )
```

### 2. **Time Series Forecasting**
```python
# Use encoder-only architecture
class TimeSeriesTransformer(nn.Module):
    def __init__(self, ...):
        # Only use encoder, not decoder
        self.encoder = nn.TransformerEncoder(...)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        encoded = self.encoder(x)
        # Predict next value
        return self.fc_out(encoded[:, -1, :])
```

### 3. **Text Classification**
```python
# Use encoder + classification head
def forward(self, src):
    encoded = self.encoder(src)
    # Pool over sequence (e.g., mean or use [CLS] token)
    pooled = encoded.mean(dim=1)
    return self.classifier(pooled)
```

## 🧠 Advanced Concepts

### Multi-Head Attention

The model uses 8 attention heads. Each head learns different patterns:
- Head 1: Might focus on adjacent tokens
- Head 2: Might focus on distant dependencies
- Head 3: Might learn syntactic patterns
- ...

**Benefits**:
- More representational capacity
- Allows model to attend to different aspects simultaneously

### Hyperparameter Tuning

Key parameters to experiment with:

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| `d_model` | Model capacity | 128-1024 |
| `nhead` | Attention diversity | 4-16 |
| `num_layers` | Model depth | 2-12 |
| `dim_feedforward` | FFN capacity | 512-4096 |
| `dropout` | Regularization | 0.1-0.3 |
| `learning_rate` | Convergence speed | 1e-5 to 1e-3 |

**Rules of thumb**:
- `d_model` must be divisible by `nhead`
- `dim_feedforward` typically 4× `d_model`
- More layers = more capacity but harder to train

### Common Issues & Solutions

#### 1. **Loss Not Decreasing**
```python
# Solution 1: Lower learning rate
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Solution 2: Add learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
```

#### 2. **Exploding Gradients**
```python
# Already implemented: gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### 3. **Slow Convergence**
```python
# Solution 1: Warmup scheduler
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
```

#### 4. **Overfitting**
```python
# Solution 1: Increase dropout
model = TransformerModel(..., dropout=0.3)

# Solution 2: Add label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Solution 3: Data augmentation
# For text: random swaps, deletions, insertions
```

### Memory Optimization

For large models:

```python
# 1. Gradient accumulation (simulate larger batch)
ACCUMULATION_STEPS = 4
for i, (src, tgt) in enumerate(dataloader):
    loss = criterion(model(src, tgt), target)
    loss = loss / ACCUMULATION_STEPS
    loss.backward()
    
    if (i + 1) % ACCUMULATION_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()

# 2. Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(src, tgt)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 3. Gradient checkpointing (trade compute for memory)
model.transformer.gradient_checkpointing_enable()
```

## 📚 Further Reading

1. **Original Paper**: "Attention Is All You Need" (Vaswani et al., 2017)
2. **Annotated Transformer**: http://nlp.seas.harvard.edu/annotated-transformer/
3. **PyTorch Docs**: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html

## 🔍 Code Structure

```
transformer_training.py
├── SequenceReversalDataset     # Data generation
├── PositionalEncoding          # Position information
├── TransformerModel            # Main model
│   ├── __init__               # Architecture setup
│   ├── forward                # Forward pass
│   └── generate_mask          # Causal masking
├── train_epoch                 # Training loop
├── evaluate                    # Validation
├── visualize_training         # Plotting
└── main                       # Orchestration
```

## 💡 Tips for Your Own Projects

1. **Start simple**: Use this sequence reversal task to verify your setup works
2. **Gradually increase complexity**: Move to real tasks once basics work
3. **Monitor training**: Watch for overfitting (train loss ↓, val loss ↑)
4. **Save checkpoints**: Don't lose hours of training!
5. **Log everything**: Use TensorBoard or Weights & Biases
6. **Validate assumptions**: Print shapes, check masks, visualize attention

Good luck with your Transformer training! 🚀