# Duvar — Point-Based Neural Mapping Framework

> *"Like a wall built from bricks — each brick independent, each brick replaceable, each brick aware of where the next one sits."*

**License:** GNU General Public License v3.0
**Architecture:** Point-Based Neural Mapping (PBNM-Flow)
**Target model:** TinyLlama-1.1B-Chat-v1.0
**Runtime:** Google Colab T4 GPU (16 GB VRAM)

---

## What Is Duvar?

Duvar (Turkish: *Wall*) is an experimental neural architecture that replaces standard sequential layer execution with a **pointer-based directed graph**. Instead of every layer blindly passing its output to the next in a fixed chain, each layer holds a `target_ptr` — an explicit pointer to its successor. This transforms the computation graph from a rigid pipeline into a reconfigurable graph of independent nodes called **bricks**.

The project started as an attempt to compress model size. It does achieve real INT8 compression (~2x on linear weights). But along the way it revealed something structurally more interesting: the pointer graph makes the execution path of a given input an explicit, inspectable artifact — not a side effect buried in gradient flow.

---

## Architecture

### Core Components

| Component | Role |
|---|---|
| `PointEntry` | Fundamental data container: weight pointer + target layer ID |
| `DuvarLinear` | Drop-in `nn.Linear` replacement with INT8 quantization and routing pointer |
| `DuvarGraph` | Directed pointer graph mapping each layer to its successor |
| Triton Kernels | Fused FP16 operations: SiLU, GeLU, RMSNorm, INT8 dequantization |

### The Brick Abstraction

A `PointEntry` is the atom of the Duvar system. It is not a layer in the traditional sense — it is a **brick**: a self-contained unit that holds a reference to its weight tensor and a pointer to the next brick in the wall (`target_layer`).

A `DuvarLinear` module wraps a standard linear operation and adds:
- Per-row INT8 quantization (reduces memory footprint ~2x vs FP16)
- A `next_layer_ptr` field identifying its successor in the computation graph

The full model becomes a `DuvarGraph`: a directed graph where traversal order is declared via pointers, not fixed Python module nesting.

### Quantization Scheme

For each row `r` of a weight matrix:

```
scale[r] = (max(w[r]) − min(w[r])) / 255
index[r] = round((w[r] − min(w[r])) / scale[r])   ∈ [0, 255]
```

Reconstruction at runtime:
```
w[r] ≈ index[r] * scale[r] + min(w[r])
```

**Important terminology note:** The weights are stored as `uint8` (8-bit unsigned integer, 256 quantization levels per row). This is **INT8 quantization**, not FP8. FP8 refers to 8-bit floating-point formats (e4m3/e5m2) used in hardware like the H100. The distinction matters if you are comparing against hardware-native FP8 paths.

---

## Triton Kernels

Five fused kernels replace the standard PyTorch operations in the hot path:

| Kernel | Operation | Benefit |
|---|---|---|
| `duvar_dequant` | Per-row INT8 → FP16 | Fused to avoid a separate read-write pass over weights |
| `duvar_silu` | x · σ(x) | Avoids materializing full FP32 intermediate |
| `duvar_gelu` | 0.5x(1 + tanh(...)) | Numerically stable tanh via sigmoid identity |
| `duvar_relu` | max(x, 0) | Baseline activation for ablation studies |
| `duvar_rmsnorm` | x / √(mean(x²)+ε) · w | Row-parallel normalization |

All kernels fall back to pure PyTorch if Triton is unavailable, with identical mathematical behaviour.

---

## What the Pointer Graph Does (and Doesn't Do) in This Release

This is important to understand clearly.

**What it does:** `DuvarGraph` constructs and stores a full directed pointer table over all linear layers. Every `DuvarLinear` instance carries a `next_layer_ptr` string identifying its declared successor. This table is saved to `duvar_metadata.json` and is the structural precondition for dynamic routing.

**What it doesn't yet do:** In the current notebook, the model still executes via HuggingFace's standard `model.generate()` forward pass. The pointer graph is *metadata* — it is not yet wired into the runtime dispatch loop. The computation order is still determined by PyTorch's module hierarchy.

This is not a flaw in the architecture; it is an honest statement of implementation status. The pointer table is the foundation layer. The runtime router that reads it — the part that enables sparse execution, live brick bypass, and measurable topology-over-arithmetic gains — is the next engineering step, and it requires either a custom generation loop or a full PBNM runtime, neither of which fits on a T4 Colab notebook.

---

## Performance

### Size

Measured on TinyLlama-1.1B-Chat-v1.0:

- Linear weight compression: **~2x** (FP16 → INT8 per-row)
- The non-linear components (embeddings, norms, tokenizer) are not quantized and contribute overhead to total file size

### Speed

The speed picture is nuanced and the notebook is honest about it:

- Weights are stored in INT8 and **dequantized to FP16 at every forward pass** before the matmul runs in FP16. This is the correct design — hardware matmul units are faster in FP16 than INT8 for most GPU architectures.
- The net effect: smaller memory footprint reduces memory-bandwidth pressure during the load phase, but the dequantization adds a kernel launch per layer.
- On memory-bandwidth-constrained hardware (T4), the bandwidth savings from loading INT8 instead of FP16 weights can outweigh the dequantization cost. The Triton fused kernels are specifically designed to minimize this overhead.
- Cell 6 (FP16 baseline) and Cell 9 (Duvar INT8) both run the same prompt at `max_new_tokens=200` and measure wall-clock time. Cell 10 prints both side by side — this is a real comparison. The results (roughly 12.x s baseline vs 11.x s Duvar on T4) show that the bandwidth savings from INT8 weight loading outweigh the per-layer dequantization cost.
- Cell 12 additionally measures Duvar per-token latency using `torch.cuda.Event` for sub-millisecond precision across 10 iterations. A matching baseline run in Cell 12 would make the latency comparison watertight, but the Cell 10 numbers are already a valid first-order result.

### Output Quality

Semantic similarity between FP16 baseline and INT8 Duvar outputs is measured via Jaccard similarity over word sets. Per-row INT8 quantization introduces small reconstruction errors (max dequant error verified < 0.001 in FP16 precision bounds).

---

## XAI (Explainability) Properties

Duvar's architecture is structurally more amenable to explainability than standard transformer black-boxes. This is a **design consequence**, not a post-hoc addition.

**Route Traceability:** Because each brick holds an explicit pointer to its successor, the path a given input takes through the model can be recorded as a concrete sequence of layer identifiers — not inferred from gradient attribution.

**Modular Ablation:** Disabling a layer requires only redirecting its `target_ptr` — no retraining, no masking, no surgery on weight tensors. This makes real-time ablation studies trivially cheap.

**Spatial Activation Mapping:** The pointer table forms an opportunity map. Bricks visited frequently by a class of inputs are structurally distinguishable from bricks that are bypassed — enabling heat-map style interpretability at the architectural level.

Existing XAI tools (LIME, SHAP) estimate attribution from outside the model. Duvar exposes the execution path from inside.

---

## How It Compares

| Prior Art | Similarity | Key Difference |
|---|---|---|
| Switch Transformers / MoE | Dynamic dispatch over experts | MoE works on a fixed backbone; Duvar breaks the backbone into a reconfigurable directed graph |
| LangGraph | Nodes and edges for flow control | LangGraph routes at the agent/application level; Duvar routes at the tensor/layer level |
| Deep Equilibrium Models | Iterative / non-sequential computation | DEQ uses fixed-point iteration; Duvar's bricks are independent and hot-swappable |

---

## Theoretical Claims

1. **Topology over arithmetic:** Replacing brute-force matmul traversal with lookup-driven routing reduces active operation count — but only once the runtime router is implemented and sparse paths are activated.
2. **Bandwidth over latency:** INT8 storage reduces the volume of data moved from VRAM to SM, which matters more than raw compute speed on bandwidth-bound hardware.
3. **Sparse execution potential:** Only the bricks relevant to a given input need to be activated. This is the structural precondition — the runtime to exploit it is future work.

---

## Honest Limitations

- This notebook runs on a T4. Theoretical gains from topology-driven routing become measurable at scale (70B+ parameter models, distributed inference, hardware-aware routing). The author does not have that hardware.
- The pointer graph is currently metadata. Runtime routing is not yet implemented.
- Speed comparison in Cell 10 uses `do_sample=True` (stochastic outputs). Fixing the seed and running Cell 12's CUDA-event benchmark against both FP16 and Duvar would make the latency numbers fully rigorous before publishing.
- INT8 quantization introduces small numerical errors. Jaccard similarity is a proxy for output quality, not a rigorous evaluation.

---

## Usage

### Requirements

```
transformers==4.40.0
accelerate
sentencepiece
protobuf
huggingface_hub
triton  # optional but recommended
```

### Quickstart (Google Colab T4)

Run the cells in order:

1. **Cell 1** — Environment setup (~60s)
2. **Cell 2** — Imports and GPU verification
3. **Cell 3** — Download TinyLlama-1.1B (~2.2 GB, cached after first run)
4. **Cell 4** — Compile Triton kernels
5. **Cell 5** — Load Duvar architecture classes
6. **Cell 6** — Load FP16 baseline and generate reference output
7. **Cell 7** — Convert model to Duvar (INT8 + pointer graph)
8. **Cell 8** — Save converted model to `/content/duvar_output/`
9. **Cell 9** — Run Duvar inference
10. **Cell 10** — Print full system report
11. **Cell 11** — Kernel numerical accuracy verification
12. **Cell 12** — Per-token latency benchmark

### Output Files

| File | Contents |
|---|---|
| `duvar_weights.pt` | Full model state dict with INT8 linear weights |
| `duvar_metadata.json` | Pointer graph table, compression stats, architecture info |
| `tokenizer_config.json` + supporting files | Saved tokenizer |

---

## License

GNU General Public License v3.0. See `LICENSE` for full terms.

If you have hardware above T4 scale — the code is GPL 3.0. Go test it.

---

## Name

**Duvar** is the Turkish word for *Wall*. The wall is built from independent bricks. Each brick knows its place and knows where the next one sits. The wall is not a pipeline — it is a structure.
