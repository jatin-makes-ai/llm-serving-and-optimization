import os
import time
import torch
import transformers.activations

# Compatibility patch for AutoAWQ with modern transformers
if not hasattr(transformers.activations, 'PytorchGELUTanh'):
    transformers.activations.PytorchGELUTanh = transformers.activations.GELUActivation

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
quant_path = "/workspace/models/qwen-0.5b-custom-awq"
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# Synthetic diverse calibration corpus across multiple domains
DOMAINS = [
    "The solar system consists of the Sun and the astronomical objects bound to it by gravity. The eight planets are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune, orbiting along elliptic planes.",
    "In computer systems, memory management involves allocating portions of physical RAM to programs at their request and freeing it when no longer needed. Modern operating systems use virtual memory addressing with page tables.",
    "Deep neural networks learn hierarchical representations through layers of linear transformations interleaved with non-linear activation functions such as ReLU, GELU, and SwiGLU.",
    "def quicksort(items):\n    if len(items) <= 1:\n        return items\n    pivot = items[len(items) // 2]\n    left = [x for x in items if x < pivot]\n    middle = [x for x in items if x == pivot]\n    right = [x for x in items if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
    "Transformer architectures eliminate recurrence by employing multi-head scaled dot-product attention, enabling extensive parallelization during training across high-performance GPUs.",
    "A database transaction enforces ACID properties: Atomicity guarantees all-or-nothing completion; Consistency ensures valid state transitions; Isolation avoids race conditions; Durability guarantees persistence.",
    "In classical economics, market equilibrium occurs where the quantity of a good demanded equals the quantity supplied, resulting in an optimal market-clearing price.",
    "Photosynthesis is the biological process used by plants, algae, and cyanobacteria to convert light energy into chemical energy stored in carbohydrate molecules like glucose."
]
calib_texts = [(DOMAINS[i % len(DOMAINS)] + f" [Sample iteration {i+1} with expanded contextual tokens describing systemic behavior and analytical metrics.]") for i in range(64)]

print("="*60)
print("      HANDS-ON MODEL QUANTIZATION: AUTOAWQ (4-BIT)")
print("="*60)
print(f"  Target Model:      {model_id}")
print(f"  Quantization Bits: {quant_config['w_bit']}-bit")
print(f"  Group Size:        {quant_config['q_group_size']}")
print(f"  Calibration Data:  {len(calib_texts)} diverse text samples")
print(f"  Output Directory:  {quant_path}")
print("="*60 + "\n")

start_t = time.time()

# Step 1: Load Unquantized Model
print("[Step 1/4] Loading unquantized FP16 weights into GPU memory...")
model = AutoAWQForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="cuda:0"
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print("           Model loaded successfully.")

# Step 2: Calibrate and search for optimal outlier scaling
print("[Step 2/4] Calibrating on diverse text samples to trace outlier activations...")
print("           Solving per-channel scaling factors (s)...")
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calib_texts,
    max_calib_samples=64,
    max_calib_seq_len=512
)

# Step 3: Pack weights into INT4 / uint32
print("[Step 3/4] Packing 4-bit weights and scales into safetensors...")
os.makedirs(quant_path, exist_ok=True)
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

elapsed = time.time() - start_t
print(f"\n[Step 4/4] SUCCESS! Model quantized and saved in {elapsed:.1f}s.")
print(f"           Artifacts stored at: {quant_path}\n")
