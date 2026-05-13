# Duvar — Point-Based Neural Mapping Framework

> *"Like a wall built from bricks — each brick independent, each brick replaceable, each brick aware of where the next one sits."*

**License:** GNU General Public License v3.0  
**Architecture:** PBNM-Flow (Point-Based Neural Mapping)  
**Reference model:** TinyLlama-1.1B-Chat-v1.0  
**Runtime:** Google Colab T4 GPU (16 GB VRAM)

---

## What Is Duvar?

Duvar (Turkish: *Wall*) is an experimental neural architecture that replaces standard sequential layer execution with a **pointer-based directed graph**. Instead of PyTorch iterating over a fixed `self.layers` list, each transformer block carries a `target_ptr` — a declared pointer to its successor. A custom forward pass reads those pointers at runtime and traverses the computation graph accordingly.

The result: the execution path of a given inference is an explicit, inspectable sequence of block identifiers. Not a side effect buried in module iteration. Not an approximation reconstructed by gradient attribution. The route is a first-class artifact of the architecture.

---

## Architecture

### Components

| Component | Role |
|---|---|
| `PointEntry` | Fundamental data unit: weight pointer + target layer ID |
| `DuvarLinear` | Drop-in `nn.Linear` replacement with per-row INT8 quantization and routing pointer |
| `DuvarGraph` | Directed pointer graph over all linear layers (compression and metadata) |
| `DuvarBlockGraph` | High-level routing graph over transformer decoder blocks (runtime control) |
| `duvar_forward()` | Custom forward function: traverses blocks via DuvarBlockGraph pointers |
| `duvar_generate()` | Custom generation loop that calls `duvar_forward` at every token step |
| Triton Kernels | Fused FP16 operations: SiLU, GeLU, RMSNorm, INT8 dequantization |

### The Brick Abstraction

A `PointEntry` is the atom of the Duvar system — a brick: a self-contained unit holding a reference to its weight tensor and a pointer to the next brick (`target_layer`).

A `DuvarLinear` wraps a standard linear operation and adds per-row INT8 quantization and a `next_layer_ptr` field. A `DuvarGraph` records the full directed pointer table over every linear layer and is serialized to `duvar_metadata.json`.

### Block-Level Routing: DuvarBlockGraph

`DuvarGraph` operates at the linear-layer level and is used for compression metadata. `DuvarBlockGraph` operates at the transformer-block level and is what drives inference.

```python
class DuvarBlockGraph:
    def __init__(self, num_layers: int):
        # Default: sequential  0 → 1 → 2 → ... → N-1 → 'OUTPUT'
        self.routing = {i: (i+1 if i+1 < num_layers else "OUTPUT")
                        for i in range(num_layers)}

    def successor(self, idx: int):
        return self.routing.get(idx, "OUTPUT")

    def set_route(self, patches: dict):
        self.routing.update(patches)  # e.g. {9: 16} skips blocks 10–15

    def active_path(self) -> list:
        path, current = [], 0
        while current != "OUTPUT":
            path.append(current)
            current = self.successor(current)
        return path
```

---

## The PBNM-Flow Forward Pass

`duvar_forward()` replaces HuggingFace's `model.generate()` and PyTorch's implicit module iteration. The traversal is explicit:

```python
def duvar_forward(model, input_ids, block_graph, route_log=None):
    hidden_states = model.model.embed_tokens(input_ids)
    position_ids  = torch.arange(seq_len, device=device).unsqueeze(0)
    causal_mask   = _build_causal_mask(seq_len, device)

    current = 0
    while current != "OUTPUT":
        layer         = model.model.layers[current]
        hidden_states = layer(hidden_states, causal_mask, position_ids)[0]
        if route_log is not None:
            route_log.append(current)          # XAI: log every block visited
        current = block_graph.successor(current)

    hidden_states = model.model.norm(hidden_states)
    return model.lm_head(hidden_states.float())
```

`duvar_generate()` wraps this into a full generation loop with top-p sampling, deterministic seeding, and a route log that records the execution path of the final token step.

---

## Routing Modes

Because the traversal order is declared in `DuvarBlockGraph`, the execution path is a runtime parameter, not a code constant.

**Sequential (default):**
`[0, 1, 2, ..., 21] → OUTPUT`
Identical computation to the standard model, verified by comparing outputs against the FP16 baseline.

**Sparse / skip routing:**
```python
sparse_graph = DuvarBlockGraph(22)
sparse_graph.set_route({9: 16})   # jump from block 9 to block 16
# active path: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21]
```
Blocks 10–15 are bypassed entirely. No retraining. No masking. No weight surgery. The change is one dictionary entry.

The route log output is concrete:
```
Route (last token): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21]
```

---

## Quantization Scheme

Per-row INT8 — not FP8. FP8 (e4m3/e5m2) is a floating-point format used in hardware like H100. Duvar stores `uint8` integers with per-row scale and min parameters.

For each row `r` of a weight matrix:

```
scale[r] = (max(w[r]) − min(w[r])) / 255
index[r] = round((w[r] − min(w[r])) / scale[r])   ∈ [0, 255]
```

Reconstruction at runtime (via fused Triton kernel):

```
w[r] ≈ index[r] * scale[r] + min(w[r])
```

Weights are dequantized to FP16 before the matmul. Hardware matmul units are faster in FP16; the INT8 format serves memory bandwidth, not arithmetic throughput.

---

## Triton Kernels

Five fused kernels replace the standard PyTorch operations in the hot path. Each falls back to a pure-PyTorch implementation (identical mathematics) if Triton is unavailable.

| Kernel | Operation | Benefit |
|---|---|---|
| `duvar_dequant` | Per-row INT8 → FP16 | Fused to avoid a separate read-write pass over weights |
| `duvar_silu` | x · σ(x) | Avoids materializing full FP32 intermediate |
| `duvar_gelu` | 0.5x(1 + tanh(...)) | Numerically stable tanh via sigmoid identity |
| `duvar_relu` | max(x, 0) | Baseline activation for ablation studies |
| `duvar_rmsnorm` | x / √(mean(x²)+ε) · w | Row-parallel normalization |

---

## Performance

### Size

Measured on TinyLlama-1.1B-Chat-v1.0:

Linear weight compression: **~2x** (FP16 → INT8 per-row). Embeddings, norms, and tokenizer are not quantized; total file compression is slightly below 2x.

### Speed

Weights are stored INT8 and dequantized to FP16 at every forward pass. On memory-bandwidth-constrained hardware (T4), the bandwidth savings from loading INT8 instead of FP16 outweigh the per-layer dequantization cost. Measured results: ~12s FP16 baseline vs ~11s Duvar (INT8 + Triton dequant) on 200-token generation.

`duvar_generate()` does not use a KV cache (straightforward to add; omitted to keep the routing logic transparent). Without a KV cache, latency grows linearly with sequence length — acceptable at demo scale.

Skip routing produces measurable additional latency reduction proportional to the number of bypassed blocks.

### Output Quality

Semantic similarity between FP16 baseline and INT8 Duvar outputs is measured via Jaccard similarity over word sets. Per-row quantization introduces small reconstruction errors; max dequant error is verified `< 0.001` in FP16 precision bounds (Cell 13).

---

## XAI (Explainability) Properties

Duvar's architecture produces explainability as a structural consequence.

**Route Traceability:** `duvar_forward()` logs every block index visited. The execution path of any inference is a concrete list of integers, not a reconstruction from gradient attribution.

**Modular Ablation:** Changing a route pointer redirects computation at the block level without modifying any weight tensor. Disabling a block costs one dictionary update.

**Spatial Activation Mapping:** Blocks visited frequently by a class of inputs are structurally distinguishable from bypassed blocks. Heat-map interpretability at the architectural level follows directly.

**Sparse Execution Proof:** `active_path()` on a `DuvarBlockGraph` returns the exact set of blocks that will be computed before inference begins. The work is declared, not inferred.

---

## Comparison with Prior Work

| Prior Art | Similarity | Key Difference |
|---|---|---|
| PathNet (DeepMind) | Path-based traversal through a super-network | PathNet uses evolutionary algorithms to find paths during training; Duvar exposes paths as a runtime parameter at inference |
| Switch Transformers / MoE | Dynamic dispatch over sub-networks | MoE operates on a fixed backbone; Duvar breaks the backbone into a hot-swappable directed graph |
| LangGraph | Nodes and edges for flow control | LangGraph routes at the agent/application level; Duvar routes at the tensor/layer level |
| Deep Equilibrium Models | Non-sequential computation | DEQ uses fixed-point iteration; Duvar's bricks are independent and the path is explicitly declared |

---

## Theoretical Claims

**Topology over arithmetic:** Replacing sequential block iteration with pointer-driven traversal enables sparse execution — only the blocks relevant to a given input need to activate. Demonstrated with `set_route({9: 16})` producing a 6-block shorter path with measurable latency reduction.

**Bandwidth over latency:** INT8 storage reduces VRAM-to-SM data movement. On bandwidth-bound hardware (T4), this is the dominant cost. Triton fused kernels minimize dequantization overhead.

**Composable routing:** Any `DuvarBlockGraph` is serializable and reversible. A route used for one input can be stored, replayed, and compared against another route — making inference topology an artifact of the same kind as a weight checkpoint.

---

## Honest Limitations

The sparse execution gains from topology-driven routing become significant at scale — 70B+ parameter models and distributed inference across many devices. TinyLlama on a T4 demonstrates the mechanism; it does not produce the magnitude of gain the architecture is designed for.

`duvar_generate()` does not implement a KV cache. This is a deliberate clarity trade-off for the demo; the pointer routing and KV cache are orthogonal and combining them is straightforward.

INT8 quantization introduces small numerical errors. Jaccard similarity is a word-overlap proxy, not a formal quality evaluation. A perplexity comparison on a held-out corpus is the correct next step.

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

### Cell Execution Order (Google Colab T4)

| Cell | Action |
|---|---|
| 1 | Environment setup (~60s) |
| 2 | Imports and GPU verification |
| 3 | Download TinyLlama-1.1B (~2.2 GB, cached after first run) |
| 4 | Compile Triton kernels |
| 5 | Load Duvar architecture classes (DuvarLinear, DuvarGraph, DuvarBlockGraph) |
| 6 | FP16 baseline inference |
| 7 | Convert model to Duvar (INT8 + pointer graphs) |
| 8 | Save converted model |
| 9 | PBNM-Flow runtime: duvar_forward + duvar_generate |
| 10 | Sequential route inference (baseline parity check) |
| 11 | Sparse route inference (skip demo + route log) |
| 12 | System report |
| 13 | Kernel numerical accuracy verification |
| 14 | Per-token latency benchmark |

### Output Files

| File | Contents |
|---|---|
| `duvar_weights.pt` | Full model state dict with INT8 linear weights |
| `duvar_metadata.json` | Linear-layer pointer graph, compression stats, architecture info |
| `tokenizer_config.json` + supporting files | Saved tokenizer |

---

## License

GNU General Public License v3.0.

If you have hardware above T4 scale — the code is GPL 3.0. Go test it.

---

## Name

**Duvar** is the Turkish word for *Wall*. The wall is built from independent bricks. Each brick knows its place and knows where the next one sits. The wall is not a pipeline — it is a structure.
