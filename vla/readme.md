# Contra-VLA: Vision-Language-Action Model for NES Contra

A Vision-Language-Action (VLA) model that fuses **text instructions**, **game screen frames**, and **RAM state** to predict future action chunks for playing NES Contra. The architecture is adapted from [SimVLA](https://github.com/SimVLA/SimVLA) (a SmolVLM-based VLA for robot manipulation) and repurposed for real-time game control.

---

## 1. Design Goals

| Goal | Decision |
|------|----------|
| **Real-time inference** | Target ≤ 50 ms / forward pass at 20 Hz agent frequency |
| **Model size** | ~300M parameters (SmolVLM-256M backbone + action transformer) |
| **Pretrained VLM** | Leverage a pretrained vision-language model for zero-shot scene understanding |
| **Action chunking** | Predict `T` future actions to reduce compounding errors and inference cost |
| **Multi-modal fusion** | Deep fusion of text, image, and structured RAM state |

---

## 2. Input Modalities

### 2.1 Text Input (Instruction / Strategy)

| Attribute | Value |
|-----------|-------|
| **Content** | Natural language task description, e.g. `"Jump over the pit and grab the spread gun"`, `"Kill the boss while dodging bullets"`, `"Stay on the lower platform to avoid enemies"` |
| **Tokenization** | AutoTokenizer from the VLM backbone (e.g. Qwen2-VL or SmolVLM) |
| **Max length** | 64 tokens (truncation / padding) |
| **Special tokens** | `<image>` placeholder token(s) inserted before or around the text, following the VLM's chat template |
| **Training source** | Human-annotated level strategies, LLM-generated tactics from RAM traces, or simple level-name templates |

Example chat template:
```
<|im_start|>user
<image>\nLevel 1: cross the bridge, watch for snipers.<|im_end|>
<|im_start|>assistant\n...
```

### 2.2 Image Input (Game Screen)

| Attribute | Value |
|-----------|-------|
| **Raw resolution** | 240 × 224 RGB (NES native) |
| **Input resolution** | **512 × 512 RGB** (resize BICUBIC — SmolVLM-256M SigLIP native resolution) |
| **Frames per step** | **2 frames** — enough to infer motion (bullets, enemies, player velocity) without excessive compute |
| **Frame sampling** | Most recent 2 frames, spaced 3 emulator frames apart (≈ 100 ms temporal gap) |
| **Color space** | RGB (preserves bullet / enemy / power-up colors critical for gameplay) |
| **Normalization** | ImageNet: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` (SmolVLM processor default) |
| **Multi-view** | No — single game screen only (unlike multi-camera robot VLAs) |

**Why 2 frames?**
- 1 frame is insufficient to infer velocity of bullets / enemies / player.
- 2 frames ≈ 100 ms of history provides motion cues (direction, speed) while keeping the VLM sequence short and inference fast.
- NES sprites move slowly enough that 2 frames are sufficient; 4 frames add diminishing returns for Contra.
- Keeps GPU memory and latency low with the 256M backbone.

> **Note on resolution:** The NES native 240×224 is upscaled to **512×512** — the native training resolution of SmolVLM-256M's SigLIP encoder (`vision_config.image_size=512`, `patch_size=16` → 32×32=1024 patches → 64 tokens after pixel-shuffle ÷4²). We do **not** use the old 168×168 resolution from `pixel2play/` — that was for the custom SmallResNet encoder. The VLM backbone replaces that entirely.

### 2.3 State Input (Structured Game State)

Unlike robot proprioception, Contra's "state" is rich and extracted from NES RAM. It is parsed by `contra/game_state.py` into a structured `proprio` vector fed directly into the action transformer.

#### Structured State Features (`proprio`) — **Canonical State Representation**
Extracted by `contra/game_state.py` into a **118-dim `float32`** vector.

| Section | Dim | Contents |
|---------|-----|----------|
| Numeric player info | 8 | scroll progress, position (x, y), velocity (vx, vy), lives, in_air |
| One-hot player info | 18 | aim direction (11) + weapon type (5) + rapid fire (2) |
| Numeric enemy info | 64 | 16 slots × (type, screen_x, screen_y, hp) — inactive slots zeroed |
| One-hot scene type | 28 | level (8) + routine (11) + location (3) + scrolling (2) + cleared/boss (4) |
| **Total** | **118** | `contra.game_state.STATE_DIM` |

The 118-dim vector is linearly projected to `hidden_size` and concatenated as a conditioning token in the action transformer (SimVLA-style `proprio`). The VLM backbone receives only text and images.

---

## 3. Network Architecture

### 3.1 High-Level Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ContraVLA Forward Pass                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Text ──→ Tokenizer ──→ Text Embeddings ──┐                                 │
│                                           ├──→ VLM Backbone ──→ VLM Features│
│  Images ──→ VLM Vision Encoder ───────────┘         [B, T_enc, D]     │     │
│                                                                        │     │
│                              ┌─────────────────────────────────────────┘    │
│                              ▼                                               │
│  Action Transformer (Causal)                                                │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  • VLM feature projection + concatenation                               │ │
│  │  • Proprio embedding (118-dim structured state → linear → hidden_size)  │ │
│  │  • Action token embeddings (learned) + positional embeddings            │ │
│  │  → Causal Transformer blocks (depth 12, hidden 768)                     │ │
│  │  → MLP head per position → 36-way class logits [B, T, 36]               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│         ↑                    │                                               │
│  Structured State            ▼                                               │
│  (118-dim float32)     Action Chunk (T actions)                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Details

#### 3.2.1 VLM Backbone

| Attribute | Value |
|-----------|-------|
| **Base model** | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| **Parameters** | ~256M |
| **Rationale** | SmolVLM-256M is the smallest variant in the SmolVLM family. It retains the same SigLIP vision encoder + Idefics3 text model architecture as SimVLA's 500M model, but at roughly half the parameter count — a sweet spot for real-time game control. |
| **Vision encoder** | SigLIP → patch features |
| **Connector** | Multi-modal projector mapping vision features to LM space |
| **Text model** | Idefics3-style lightweight LLM |
| **Hidden size `D`** | 576 |
| **Frozen at start?** | Yes — first `N` warmup steps freeze VLM, train only action transformer + heads |
| **Differential LR** | VLM uses `lr * 0.1`, action transformer uses full `lr` |

**Efficient forward pass**: During training we bypass the chat template and directly construct `[image_tokens, text_tokens]` sequences for batch efficiency, similar to SimVLA's `forward_vlm_efficient`.

#### 3.2.2 Action Transformer

A standard causal Transformer decoder that takes the VLM's fused vision-language features plus structured state, and auto-regressively predicts action tokens.

| Attribute | Value |
|-----------|-------|
| **Hidden size** | 768 (projected from VLM's `D=576` via a linear layer) |
| **Depth** | 12 layers |
| **Num heads** | 12 |
| **MLP ratio** | 4× |
| **Positional encoding** | Learned 1D positional embeddings on action tokens |
| **Attention mask** | Causal (action token `i` attends to tokens `< i`) |

**Input construction:**
1. **Action token embeddings**: Ground-truth action indices are embedded via `nn.Embedding(36, hidden_size)` during training; auto-regressively sampled during inference.
2. **Positional embeddings**: Learned 1D positions for each action slot in the chunk.
3. **Proprio embedding**: 118-dim structured state → linear projection to `hidden_size`.
4. **VLM condition**: VLM output features are mean-pooled → linear projection to `hidden_size`.
5. **Concatenate sequence**: `x = [vlm_proj(vlm_features), proprio_emb, action_emb_0, action_emb_1, ..., action_emb_{T-1}]`
6. **Causal Transformer**: Pre-LN self-attention blocks with causal masking. Each action position can attend to the VLM/state prefix and all prior action positions.
7. **Output head**: Shared MLP classifier head on each action position → logits over 36 classes.

#### 3.2.3 Action Head — Discrete Action Space

> **Critical difference from SimVLA:** SimVLA predicts continuous robot joint angles with flow matching. **Contra has a discrete action space** (36 classes). We therefore use a **categorical prediction head** rather than a continuous regression head.

| Attribute | Value |
|-----------|-------|
| **Action space** | **36 discrete classes** = 9 D-pad directions × 4 button combos |
| **D-pad** | `_`, `L`, `R`, `U`, `D`, `UL`, `UR`, `DL`, `DR` |
| **Buttons** | `_`, `J` (jump/A), `F` (fire/B), `FJ` |
| **Action chunk size `T`** | **8 future actions** (~400 ms at 20 Hz) |
| **Prediction target** | Categorical distribution over 36 classes per timestep |

##### Primary Mode: Causal BC with Cross-Entropy (Recommended)

This is the simplest, most proven approach for discrete actions and aligns with the existing `pixel2play/` transformer.

**Sequence layout:**
```
[image_tokens_view0] [image_tokens_view1] [text_tokens] [state_token] [a_0] [a_1] ... [a_{T-1}]
```

**Causal mask rules:**
- Image, text, and state tokens are fully bidirectional (can attend to each other).
- Action token `a_i` can attend to all image/text/state tokens **and** all previous action tokens `a_0 ... a_{i-1}`.
- Action token `a_i` **cannot** attend to future action tokens `a_{i+1} ...`.

**Training:** Teacher-forced cross-entropy:
```python
loss = CrossEntropy(logits[:, i, :], action_labels[:, i])  # for i = 0..T-1
```

**Inference:** Auto-regressive sampling (greedy or temperature):
```python
for i in range(T):
    logits = model(..., generated_actions[:i])
    action_i = argmax(logits)  # or sample with temperature
```

##### Experimental Mode: Discrete Flow Matching

As a research extension, we can adapt flow matching to discrete actions by operating in **learned action embedding space**:

1. Embed action indices `a` → `e_a ∈ R^d` via a learned `nn.Embedding(36, d)`.
2. Sample `t ~ Beta(1.5, 1)`.
3. Interpolate: `x_t = t * noise + (1 - t) * e_a`.
4. Model predicts denoising velocity `v_t`.
5. Loss: `MSE(v_pred, noise - e_a)`.
6. Inference: Euler integration from noise → `x_0` → nearest-neighbor lookup in embedding table → action index.

**Status:** Experimental only. Causal BC is the default because it is stable, fast (single forward pass), and matches the existing project's proven pipeline.

---

## 4. Output: Action Chunk

| Attribute | Value |
|-----------|-------|
| **Chunk size** | 8 actions |
| **Temporal spacing** | Every action is executed for 3 emulator frames (1 agent step = 50 ms) |
| **Chunk duration** | 8 × 50 ms = 400 ms of gameplay |
| **Execution** | Execute action 0 immediately, then action 1, ... up to action 7 |
| **Replanning** | After executing 8 actions (or every 4 actions with temporal aggregation), run a new forward pass |
| **Temporal aggregation** | Optional: execute first 4 actions, then replan — reduces latency while keeping chunk benefits |

**Action post-processing:**
- The model predicts 36-class logits per timestep.
- During training: cross-entropy with ground-truth human / search traces.
- During inference: argmax (greedy) or temperature sampling.
- The 36-class index is decoded back to NES `MultiBinary(9)` via `pixel2play.dataset.combine_dpad_button`.
- **No action normalization** is needed (unlike SimVLA's continuous robot actions) because our action space is already discrete and bounded.

---

## 5. Pretrained Checkpoints

### 5.1 Loading Pretrained VLM Weights

```python
from transformers import AutoModelForVision2Seq, AutoProcessor

# Load VLM backbone
vlm_backbone = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")

# Initialize ContraVLA with VLM weights
model = ContraVLA(
    vlm_config=vlm_backbone.config,
    action_dim=36,
    num_actions=8,
    hidden_size=768,
    num_layers=12,
)
model.vlm.load_state_dict(vlm_backbone.state_dict(), strict=False)
```

### 5.2 Checkpoint Format

The model is a HuggingFace `PreTrainedModel` with a custom config:

```python
class ContraVLAConfig(PretrainedConfig):
    model_type = "contravla"
    
    def __init__(
        self,
        vlm_model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
        action_dim=36,
        num_actions=8,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        # Note: only causal transformer is supported; DiT/AdaLN removed
        proprio_dim=118,     # structured state dimension (game_state.STATE_DIM)
        ram_dim=2048,        # optional raw RAM tokenization
        dropout=0.1,
        **kwargs,
    ):
        ...
```

Saved checkpoints contain:
```
checkpoint/
├── config.json              # ContraVLAConfig
├── model.safetensors        # All weights (VLM + action transformer + heads)
├── preprocessor_config.json # Processor / tokenizer config
└── action_stats.json        # Action normalization stats (if any)
```

### 5.3 Training Schedule with Pretrained Weights

| Phase | Steps | VLM LR | Action LR | Notes |
|-------|-------|--------|-----------|-------|
| **Warmup (freeze VLM)** | 0 – 1,000 | 0.0 | 1e-4 | Only action transformer + heads train |
| **Full fine-tuning** | 1,000 – 50,000 | 1e-5 | 1e-4 | VLM unfrozen, 0.1× LR |
| **Cosine decay** | 50,000 – 100,000 | 1e-5 → 1e-6 | 1e-4 → 1e-5 | Gradual decay |

---

## 6. Dataset & Training

### 6.1 Training Data Format

Each training sample is a dict:

```python
{
    "input_ids": Tensor[B, L],              # Tokenized text instruction
    "images": Tensor[B, V, C, H, W],        # V=2 frames, C=3, H=W=384
    "image_mask": Tensor[B, V],             # All True for our use case
    "proprio": Tensor[B, 118],               # Structured state (game_state.STATE_DIM)
    "actions": Tensor[B, T],                # Action chunk indices [0..35]
}
```

Data sources:
1. **Human recordings** (`contra/human_recordings/`) — expert demonstrations
2. **Monte Carlo search traces** (`synthetic/mc_trace/`) — synthetic winning trajectories
3. **Pruned traces** — after `trim_fire_actions.py` removes redundant inputs

### 6.2 Data Augmentation

| Modality | Augmentation |
|----------|-------------|
| **Images** | RandomColorJitter (brightness, contrast, saturation), random translation ±4 px |
| **Text** | Random template substitution (e.g. replace "spread gun" with "S gun", "power-up") |
| **Actions** | None (preserve exact human timing) |

### 6.3 Loss Function

**Primary loss (causal BC):**
```python
loss = CrossEntropy(action_logits, action_labels)  # per-token, teacher-forced
```

**Optional auxiliary losses:**
```python
loss_total = loss_bc + 0.1 * loss_vlm_prefix  # if using VLM text generation as aux task
```
---

## 9. File Structure (Proposed)

```
vla/
├── readme.md                          # This document
├── model/
│   ├── __init__.py
│   ├── configuration_contravla.py     # HF Config class
│   ├── modeling_contravla.py          # Main ContraVLA model
│   ├── action_transformer.py          # Causal action transformer
│   ├── action_hub.py                  # Action normalization, encoding, decoding
│   └── processing_contravla.py        # Image + text preprocessor
├── train.py                           # Training loop with Accelerate
├── inference.py                       # Standalone inference script
├── eval_env.py                        # Evaluation in stable-retro
└── datasets/
    ├── __init__.py
    └── contra_vla_dataset.py          # PyTorch dataset from npz shards
```

---

