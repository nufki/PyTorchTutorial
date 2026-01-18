# Transformer Hyperparameters - Complete Guide

## Overview
This guide explains every parameter in the Transformer training configuration, how they affect your model, and how to tune them for different scenarios.

```python
VOCAB_SIZE = 20
SEQ_LEN = 10
D_MODEL = 128
NHEAD = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001
```

---

## Data Parameters

### `VOCAB_SIZE = 20`
**What it is:** The number of unique tokens (words, characters, or numbers) in your vocabulary.

**In this example:** Numbers 0-19 (we use 1-19 for data, 0 is reserved for padding)

**How it affects the model:**
- Determines input/output embedding layer size
- More vocab = larger embedding tables = more parameters

**Real-world examples:**
```python
VOCAB_SIZE = 20           # This toy example (numbers 0-19)
VOCAB_SIZE = 50           # Small alphabet + punctuation
VOCAB_SIZE = 50_000       # GPT-2 (subword tokens)
VOCAB_SIZE = 100_000      # Large language models
```

**Memory impact:**
- Embedding layer: `VOCAB_SIZE × D_MODEL` parameters
- Output layer: `D_MODEL × VOCAB_SIZE` parameters
- Example: 20 × 128 × 2 = 5,120 parameters

**When to change:**
- ✅ Based on your data (number of unique tokens)
- ❌ Don't change arbitrarily - it's determined by your dataset

---

### `SEQ_LEN = 10`
**What it is:** Maximum sequence length the model can process at once.

**In this example:** Each input sequence is 10 numbers long

**How it affects the model:**
- Defines the context window
- Longer sequences = more memory usage (quadratic with attention!)
- Affects positional encoding size

**Trade-offs:**
```python
SEQ_LEN = 10     # Fast, less memory, limited context
SEQ_LEN = 50     # Moderate (good for sentences)
SEQ_LEN = 512    # Standard (BERT, GPT-2)
SEQ_LEN = 2048   # Large (GPT-3)
SEQ_LEN = 100000 # Massive (modern LLMs with special attention)
```

**Memory complexity:** O(SEQ_LEN²) due to attention mechanism
- 10 tokens: 100 attention computations
- 100 tokens: 10,000 attention computations (100x more!)
- 1000 tokens: 1,000,000 attention computations

**When to change:**
- Short sequences (≤50): Classification, simple tasks
- Medium (50-512): Translation, summarization
- Long (512+): Document understanding, long-form generation

**💡 Tip:** Start small, increase if model can't capture dependencies

---

## Model Architecture Parameters

### `D_MODEL = 128`
**What it is:** The dimensionality of the model's internal representations (embedding size).

**How it works:**
- All vectors inside the Transformer have this dimension
- Embeddings: tokens → 128-dim vectors
- Attention: queries, keys, values are all 128-dim
- Each layer processes 128-dim vectors

**Think of it as:** The "width" or "capacity" of the model

**Common values:**
```python
D_MODEL = 64      # Tiny (toy examples)
D_MODEL = 128     # Small (this example)
D_MODEL = 256     # Moderate (small real tasks)
D_MODEL = 512     # Standard (original Transformer paper)
D_MODEL = 768     # BERT-base
D_MODEL = 1024    # Large models
D_MODEL = 4096    # GPT-3
D_MODEL = 12288   # GPT-4 (estimated)
```

**Memory impact:**
- Most parameters scale with D_MODEL²
- Doubling D_MODEL ≈ 4x parameters

**Performance:**
```
D_MODEL = 64   → Can learn simple patterns
D_MODEL = 256  → Can learn moderate complexity
D_MODEL = 1024 → Can learn very complex patterns
```

**When to change:**
- Increase if: Model underfitting, task is complex
- Decrease if: Out of memory, task is simple

**⚠️ Constraint:** Must be divisible by `NHEAD`
```python
D_MODEL = 128, NHEAD = 8  # ✅ 128/8 = 16 (valid)
D_MODEL = 100, NHEAD = 8  # ❌ 100/8 = 12.5 (invalid!)
```

---

### `NHEAD = 8`
**What it is:** Number of parallel attention heads in multi-head attention.

**How it works:**
Each head learns different aspects:
```
Head 1: Might focus on nearby tokens (local dependencies)
Head 2: Might focus on distant tokens (long-range dependencies)
Head 3: Might focus on syntactic relationships
Head 4: Might focus on semantic relationships
...and so on
```

**Mechanism:**
```python
# D_MODEL is split across heads
D_MODEL = 128
NHEAD = 8
head_dim = D_MODEL // NHEAD = 16  # Each head gets 16 dimensions

# Attention is computed in parallel for each head, then concatenated
```

**Common values:**
```python
NHEAD = 1   # Single-head (simple attention)
NHEAD = 4   # Small
NHEAD = 8   # Standard (this example, BERT, GPT-2)
NHEAD = 12  # BERT-large
NHEAD = 16  # Large models
NHEAD = 32  # Very large models
```

**Trade-offs:**
- **More heads:**
  - ✅ More diverse attention patterns
  - ✅ Better at capturing different types of relationships
  - ❌ More computation
  - ❌ Smaller dimension per head (might limit expressiveness)

- **Fewer heads:**
  - ✅ Less computation
  - ✅ Larger dimension per head
  - ❌ Less diversity in attention patterns

**Typical ratios:**
```python
D_MODEL = 512,  NHEAD = 8   # head_dim = 64  (original Transformer)
D_MODEL = 768,  NHEAD = 12  # head_dim = 64  (BERT)
D_MODEL = 1024, NHEAD = 16  # head_dim = 64  (common pattern)
```

**💡 Sweet spot:** `head_dim` between 32-128 works well

**When to change:**
- Increase if: Model struggles with different types of dependencies
- Decrease if: Out of memory, simpler tasks

---

### `NUM_ENCODER_LAYERS = 3`
**What it is:** Number of stacked encoder layers.

**How it works:**
```
Input Sequence
    ↓
[Encoder Layer 1]  ← Self-attention + Feedforward
    ↓
[Encoder Layer 2]  ← Self-attention + Feedforward
    ↓
[Encoder Layer 3]  ← Self-attention + Feedforward
    ↓
Encoded Representation
```

Each layer refines the representation:
- Layer 1: Basic patterns (nearby tokens)
- Layer 2: Intermediate patterns (phrases, local context)
- Layer 3: High-level patterns (sentence structure, meaning)

**Common values:**
```python
NUM_ENCODER_LAYERS = 1   # Minimal (rarely used)
NUM_ENCODER_LAYERS = 3   # Small (this example)
NUM_ENCODER_LAYERS = 6   # Standard (original Transformer, BERT-base)
NUM_ENCODER_LAYERS = 12  # Large (BERT-large, GPT-2)
NUM_ENCODER_LAYERS = 24  # Very large (GPT-3)
NUM_ENCODER_LAYERS = 96  # Massive (GPT-4, estimated)
```

**Depth vs. Width:**
```
Deeper (more layers):
  ✅ Better at hierarchical abstractions
  ✅ Learns more complex patterns
  ❌ Slower training
  ❌ Risk of gradient vanishing

Wider (larger D_MODEL):
  ✅ More capacity per layer
  ✅ Faster to train (fewer sequential steps)
  ❌ May not capture hierarchical structure as well
```

**Rule of thumb:**
- **Simple tasks**: 2-4 layers
- **Moderate tasks**: 6-8 layers
- **Complex tasks**: 12+ layers

**When to change:**
- Increase if: Model underfitting, complex reasoning needed
- Decrease if: Overfitting, training too slow, simple task

---

### `NUM_DECODER_LAYERS = 3`
**What it is:** Number of stacked decoder layers.

**How it works:**
```
Target Sequence (shifted)
    ↓
[Decoder Layer 1]  ← Masked self-attention + Cross-attention + Feedforward
    ↓
[Decoder Layer 2]  ← Masked self-attention + Cross-attention + Feedforward
    ↓
[Decoder Layer 3]  ← Masked self-attention + Cross-attention + Feedforward
    ↓
Output Predictions
```

**Typically same as encoder layers:**
```python
# Balanced (most common)
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6

# Encoder-heavy (sometimes used for understanding-focused tasks)
NUM_ENCODER_LAYERS = 12
NUM_DECODER_LAYERS = 6

# Decoder-only models (GPT-style)
NUM_ENCODER_LAYERS = 0
NUM_DECODER_LAYERS = 24
```

**Common configurations:**
```python
Transformer (original):  6 encoder + 6 decoder
BERT:                   12 encoder + 0 decoder  (encoder-only)
GPT-2:                   0 encoder + 12 decoder (decoder-only)
T5:                     12 encoder + 12 decoder
```

**When to change:**
- Usually keep same as `NUM_ENCODER_LAYERS`
- Increase decoder if generation quality is poor
- Decrease decoder if input understanding is more important

---

### `DIM_FEEDFORWARD = 512`
**What it is:** Hidden dimension of the position-wise feedforward network inside each Transformer layer.

**How it works:**
After attention, each position goes through a 2-layer MLP:
```python
# Inside each Transformer layer:
x = attention(x)           # D_MODEL → D_MODEL
x = feedforward(x)         # D_MODEL → DIM_FEEDFORWARD → D_MODEL

# Feedforward network:
def feedforward(x):
    x = Linear(D_MODEL → DIM_FEEDFORWARD)  # Expand
    x = ReLU(x)                            # Non-linearity
    x = Linear(DIM_FEEDFORWARD → D_MODEL)  # Compress
    return x
```

**Think of it as:** The "processing power" at each position

**Common ratios:**
```python
# Standard: 4x the model dimension
D_MODEL = 128,  DIM_FEEDFORWARD = 512   # 4x (this example)
D_MODEL = 512,  DIM_FEEDFORWARD = 2048  # 4x (original Transformer)
D_MODEL = 768,  DIM_FEEDFORWARD = 3072  # 4x (BERT)

# Sometimes different ratios:
D_MODEL = 1024, DIM_FEEDFORWARD = 4096  # 4x
D_MODEL = 1024, DIM_FEEDFORWARD = 8192  # 8x (more capacity)
```

**Memory impact:**
- Feedforward parameters: `2 × D_MODEL × DIM_FEEDFORWARD`
- Often the largest component (~2/3 of total parameters)

**Example calculation:**
```python
D_MODEL = 128
DIM_FEEDFORWARD = 512
Parameters per layer = 2 × 128 × 512 = 131,072 parameters
```

**Trade-offs:**
- **Larger DIM_FEEDFORWARD:**
  - ✅ More processing capacity per position
  - ✅ Can learn more complex transformations
  - ❌ More parameters
  - ❌ More memory usage

**When to change:**
- Increase if: Model underfitting, have memory available
- Decrease if: Out of memory, overfitting
- Keep at 4× D_MODEL as a default

---

## Regularization Parameters

### `DROPOUT = 0.1`
**What it is:** Probability of randomly "dropping out" (setting to zero) neurons during training.

**How it works:**
```python
# During training:
x = some_layer(x)
x = dropout(x, p=0.1)  # Randomly zero out 10% of values

# During inference:
# Dropout is disabled (all neurons active)
```

**Purpose:** Prevents overfitting by:
- Forcing network to learn redundant representations
- Making it not rely on any single neuron
- Acts as an ensemble of smaller networks

**Common values:**
```python
DROPOUT = 0.0   # No dropout (might overfit)
DROPOUT = 0.1   # Light regularization (this example, BERT)
DROPOUT = 0.2   # Moderate regularization
DROPOUT = 0.3   # Strong regularization
DROPOUT = 0.5   # Very strong (classic CNNs, rarely for Transformers)
```

**Guidelines:**
```python
# Small dataset
DROPOUT = 0.2-0.3  # Need more regularization

# Large dataset
DROPOUT = 0.1      # Less regularization needed

# If overfitting (train loss << val loss)
DROPOUT = 0.2-0.4  # Increase dropout

# If underfitting (both losses high)
DROPOUT = 0.0-0.1  # Decrease dropout
```

**Where it's applied in Transformers:**
- Attention weights
- After attention output
- In feedforward network
- In embedding layers

**When to change:**
- Increase if: Overfitting (train acc >> val acc)
- Decrease if: Underfitting (both accuracies low)
- Set to 0 if: Dataset is huge, using other regularization

---

## Training Parameters

### `BATCH_SIZE = 64`
**What it is:** Number of examples processed together in one forward/backward pass.

**How it works:**
```python
# Instead of:
for each_example in dataset:  # One at a time
    loss = compute_loss(example)
    loss.backward()

# We do:
for batch in dataset:  # 64 examples at once
    loss = compute_loss(batch)
    loss.backward()  # Update based on batch average
```

**Trade-offs:**

**Larger batch (64, 128, 256):**
- ✅ Faster training (GPU utilization)
- ✅ More stable gradients
- ✅ Better parallelization
- ❌ More memory required
- ❌ May converge to sharper minima (worse generalization)
- ❌ Need to lower learning rate

**Smaller batch (8, 16, 32):**
- ✅ Less memory
- ✅ More gradient noise (can help escape local minima)
- ✅ Better generalization sometimes
- ❌ Slower training
- ❌ More unstable gradients
- ❌ Noisy loss curves

**Common values by scenario:**
```python
# Personal laptop (8-16GB RAM)
BATCH_SIZE = 8-16

# Workstation (32GB RAM, good GPU)
BATCH_SIZE = 32-64    # This example

# High-end GPU (A100, 80GB)
BATCH_SIZE = 128-512

# Training large models (distributed)
BATCH_SIZE = 1024-4096 (effective, via gradient accumulation)
```

**Memory calculation:**
```python
Memory per batch ≈ BATCH_SIZE × SEQ_LEN × D_MODEL × 4 bytes
Example: 64 × 10 × 128 × 4 = 327,680 bytes ≈ 320KB (just activations)
```

**Relationship with learning rate:**
```python
# Rule of thumb: Linear scaling
BATCH_SIZE = 32,  LEARNING_RATE = 0.0001
BATCH_SIZE = 64,  LEARNING_RATE = 0.0002  # 2x batch, 2x LR
BATCH_SIZE = 128, LEARNING_RATE = 0.0004  # 4x batch, 4x LR
```

**When to change:**
- Increase if: Have more memory, want faster training
- Decrease if: Out of memory, want better generalization
- Use gradient accumulation if you want large effective batch size but lack memory

---

### `NUM_EPOCHS = 50`
**What it is:** Number of complete passes through the entire training dataset.

**What one epoch means:**
```python
dataset_size = 5000 examples
BATCH_SIZE = 64

batches_per_epoch = 5000 / 64 ≈ 78 batches
NUM_EPOCHS = 50

Total training steps = 50 × 78 = 3,900 steps
```

**How to choose:**

**Underfitting symptoms (need MORE epochs):**
- Training loss still decreasing at end
- Validation loss still decreasing
- Model hasn't converged
```python
NUM_EPOCHS = 100-200  # Train longer
```

**Overfitting symptoms (need FEWER epochs):**
- Training loss decreasing, validation loss increasing
- Large gap between train and val accuracy
```python
NUM_EPOCHS = 20-30    # Stop earlier
# Or use early stopping
```

**Common values by task:**
```python
# Simple tasks (like sequence reversal)
NUM_EPOCHS = 20-50    # This example

# Moderate tasks (classification)
NUM_EPOCHS = 50-100

# Complex tasks (large language models)
NUM_EPOCHS = 1-3      # One pass through massive data is enough!

# Small datasets
NUM_EPOCHS = 100-500  # Need many passes
```

**💡 Best practice:** Use early stopping instead of fixed epochs
```python
# Stop when validation loss hasn't improved for 10 epochs
patience = 10
best_val_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(MAX_EPOCHS):
    val_loss = train_and_evaluate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        save_model()
    else:
        epochs_without_improvement += 1
    
    if epochs_without_improvement >= patience:
        print("Early stopping!")
        break
```

**When to change:**
- Increase if: Model still improving
- Decrease if: Model overfitting early
- Use early stopping for automatic tuning

---

### `LEARNING_RATE = 0.0001`
**What it is:** Step size for weight updates during gradient descent.

**How it works:**
```python
# Gradient descent update rule:
weight = weight - LEARNING_RATE × gradient

# Example:
gradient = 0.5
weight = 1.0

# Small LR (0.0001):
weight = 1.0 - 0.0001 × 0.5 = 0.99995  # Tiny step

# Large LR (0.1):
weight = 1.0 - 0.1 × 0.5 = 0.95        # Big step
```

**The Goldilocks problem:**

**Too small (0.00001):**
- ❌ Training extremely slow
- ❌ May get stuck in local minima
- ❌ Takes forever to converge
```
Epoch 1: loss = 2.5
Epoch 100: loss = 2.4  # Barely moved!
```

**Too large (0.01):**
- ❌ Training unstable
- ❌ Loss oscillates or explodes
- ❌ May never converge
```
Epoch 1: loss = 2.5
Epoch 2: loss = 150.3  # Exploded!
```

**Just right (0.0001-0.001):**
- ✅ Steady decrease
- ✅ Stable training
- ✅ Converges in reasonable time
```
Epoch 1: loss = 2.5
Epoch 10: loss = 1.2
Epoch 50: loss = 0.05  # Nice!
```

**Common values by optimizer:**
```python
# Adam (this example) - generally higher LR
LEARNING_RATE = 0.0001-0.001  # Default: 0.001

# AdamW (with weight decay)
LEARNING_RATE = 0.00001-0.0001  # Slightly lower

# SGD - needs much higher LR
LEARNING_RATE = 0.01-0.1

# SGD with momentum
LEARNING_RATE = 0.001-0.01
```

**Learning rate schedules:**

**1. Constant (simplest):**
```python
LR = 0.0001  # Same throughout training
```

**2. Step decay:**
```python
# Reduce by 10x every 30 epochs
Epoch 0-29:   LR = 0.001
Epoch 30-59:  LR = 0.0001
Epoch 60+:    LR = 0.00001
```

**3. Warmup + Decay (Transformer standard):**
```python
# Linear warmup, then decay
Epoch 0-10:   LR increases from 0 to 0.001
Epoch 11-50:  LR decreases from 0.001 to 0.0001
```

**4. Cosine annealing:**
```python
# Smooth cosine curve from high to low
LR = 0.001 × 0.5 × (1 + cos(π × epoch / total_epochs))
```

**5. ReduceLROnPlateau (adaptive):**
```python
# Reduce LR when validation loss plateaus
if val_loss hasn't improved for 5 epochs:
    LR = LR × 0.5
```

**How to tune:**

**Start with:** `1e-3` (0.001) for Adam

**If loss explodes:**
```python
LEARNING_RATE = 0.0001  # Reduce 10x
```

**If learning too slow:**
```python
LEARNING_RATE = 0.001   # Increase 10x
```

**Quick tuning experiment:**
```python
# Try these and pick the best:
learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]

for lr in learning_rates:
    model = create_model()
    train_for_10_epochs(model, lr)
    print(f"LR {lr}: final_loss = {loss}")

# Pick the one with lowest loss
```

**💡 Learning rate finder:**
```python
# Increase LR exponentially, plot loss
lrs = []
losses = []

lr = 1e-7
for batch in first_few_hundred_batches:
    loss = train_step(batch, lr)
    lrs.append(lr)
    losses.append(loss)
    lr *= 1.1  # Increase by 10% each step

# Plot: loss vs lr
# Pick LR just before loss starts increasing
```

**When to change:**
- Decrease if: Loss exploding, NaN values
- Increase if: Loss decreasing too slowly
- Use schedule if: Training for many epochs

---

## Parameter Interactions & Rules

### Critical Constraints:

**1. D_MODEL must be divisible by NHEAD:**
```python
✅ D_MODEL = 128, NHEAD = 8   # 128 / 8 = 16
✅ D_MODEL = 512, NHEAD = 8   # 512 / 8 = 64
❌ D_MODEL = 100, NHEAD = 8   # 100 / 8 = 12.5 (ERROR!)
```

**2. Batch size × Sequence length must fit in memory:**
```python
# Rough estimate:
memory_GB = BATCH_SIZE × SEQ_LEN × D_MODEL × num_layers × 4 / 1e9

# This example:
64 × 10 × 128 × 6 × 4 / 1e9 ≈ 0.002 GB (2MB) - tiny!

# Large model:
128 × 512 × 1024 × 24 × 4 / 1e9 ≈ 6.4 GB - significant
```

### Common Presets:

**Tiny (debugging, toy examples):**
```python
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
BATCH_SIZE = 32
```

**Small (this example):**
```python
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 3
DIM_FEEDFORWARD = 512
BATCH_SIZE = 64
```

**BERT-base:**
```python
D_MODEL = 768
NHEAD = 12
NUM_ENCODER_LAYERS = 12
DIM_FEEDFORWARD = 3072
BATCH_SIZE = 32
SEQ_LEN = 512
```

**GPT-2:**
```python
D_MODEL = 768
NHEAD = 12
NUM_DECODER_LAYERS = 12
DIM_FEEDFORWARD = 3072
SEQ_LEN = 1024
```

**GPT-3 (175B):**
```python
D_MODEL = 12288
NHEAD = 96
NUM_DECODER_LAYERS = 96
DIM_FEEDFORWARD = 49152
SEQ_LEN = 2048
```

---

## Quick Reference Table

| Parameter | This Example | Typical Range | Primary Effect |
|-----------|--------------|---------------|----------------|
| VOCAB_SIZE | 20 | 1K-100K | Dataset-dependent |
| SEQ_LEN | 10 | 10-4096 | Context window, memory |
| D_MODEL | 128 | 64-12288 | Model capacity |
| NHEAD | 8 | 4-96 | Attention diversity |
| NUM_ENCODER_LAYERS | 3 | 1-96 | Encoding depth |
| NUM_DECODER_LAYERS | 3 | 1-96 | Generation depth |
| DIM_FEEDFORWARD | 512 | 256-49152 | Processing power |
| DROPOUT | 0.1 | 0.0-0.5 | Regularization |
| BATCH_SIZE | 64 | 8-4096 | Speed vs. memory |
| NUM_EPOCHS | 50 | 10-500 | Training duration |
| LEARNING_RATE | 0.0001 | 1e-5 to 1e-2 | Convergence speed |

---

## Tuning Workflow

### Step 1: Start with defaults
```python
# Use the values from this example
# They're reasonable for small tasks
```

### Step 2: Monitor training
```python
# Watch for these signals:
- Loss decreasing? ✅ Good
- Loss flat? → Increase learning rate or model size
- Loss exploding? → Decrease learning rate
- Train << Val loss? → Increase dropout or reduce model size
```

### Step 3: Adjust one at a time
```python
# Don't change everything at once!
# Change one parameter, retrain, evaluate
```

### Step 4: Scale up gradually
```python
# If model learns well but needs more capacity:
D_MODEL: 128 → 256
NUM_LAYERS: 3 → 6
DIM_FEEDFORWARD: 512 → 1024
```

Happy training! 🚀
