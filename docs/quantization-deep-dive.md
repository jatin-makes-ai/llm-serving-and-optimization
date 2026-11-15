# Deep-Dive: LLM Quantization Mathematics & Hardware Execution

A comprehensive technical reference detailing how 4-bit and 8-bit quantization works in silicon, bit-packing and unpacking mechanics, and the mathematical differences between AWQ, GPTQ, and FP8.

---

## 1. Why Naive Downsizing (Round-to-Nearest) Destroys an LLM

In standard software engineering, converting a `float` to an `int` is just truncation or rounding (`int(x)`). In Large Language Models, doing naive rounding (Round-to-Nearest / RTN) causes output perplexity to explode to infinity (producing complete gibberish).

### The Activation Outlier Problem
In modern Transformers:
* Across $\approx 99.9\%$ of the network, activations are small, normally distributed values between $[-1.0, 1.0]$.
* However, in a tiny fraction of hidden channels ($\approx 0.1\%$), activations exhibit **emergent outlier spikes** with magnitudes reaching $+50.0$ to $+100.0$ across all sequence tokens.
* If a weight connected to an outlier channel is rounded by just $0.05$, that rounding error is multiplied by $100.0$, introducing an error of $+5.0$ that cascades through all 28+ Transformer layers, corrupting attention and MLP outputs.

### The Quantization Objective
All modern post-training quantization methods formulate quantization as an optimization problem:

$$\min_{\hat{W}} \| W X - \hat{W} X \|_2^2$$

We search for a discrete quantized matrix $\hat{W}$ such that when multiplied by realistic activation inputs $X$, the output $\hat{W} X$ matches the unquantized 16-bit output $W X$ with minimal divergence.

---

## 2. A Step-by-Step Numerical Example: Unpacking INT4 to FP16

Here is the exact journey of two neural network weights from **original float** $\rightarrow$ **packed 4-bit byte in VRAM** $\rightarrow$ **unpacked back to FP16 in GPU registers**.

### Step 1: The Original 16-bit Float Weights in PyTorch
In layer 14 of an LLM, consider two adjacent weights in a group of 128 weights:
* **Weight 1 ($W_1$)**: `-0.125`
* **Weight 2 ($W_2$)**: `+0.200`

In 16-bit float (`FP16`), each weight takes 2 bytes (16 bits). Together, they take **4 bytes (32 bits)** in VRAM.

---

### Step 2: The Offline Quantizer Calculates Group Scale ($S$) & Zero-Point ($Z$)
For this group of 128 weights, suppose the minimum value is `-0.32` and the maximum is `+0.28`.
A 4-bit unsigned integer can hold values from **0 to 15** ($2^4 = 16$ discrete bins).

1. **Scale ($S$)**: 
   $$S = \frac{\text{Max} - \text{Min}}{15} = \frac{0.28 - (-0.32)}{15} = \frac{0.60}{15} = \mathbf{0.04}$$
2. **Zero-Point ($Z$)**:
   $$Z = \text{round}\left(-\frac{\text{Min}}{S}\right) = \text{round}\left(-\frac{-0.32}{0.04}\right) = \mathbf{8}$$

Now we quantize the two weights into 4-bit integers ($q$):
* **Quantize $W_1$**:
  $$q_1 = \text{round}\left(\frac{-0.125}{0.04}\right) + 8 = -3 + 8 = \mathbf{5} \quad (\text{Binary: } \mathbf{0101}_2)$$
* **Quantize $W_2$**:
  $$q_2 = \text{round}\left(\frac{+0.200}{0.04}\right) + 8 = +5 + 8 = \mathbf{13} \quad (\text{Binary: } \mathbf{1101}_2)$$

---

### Step 3: Packing into a Single 8-bit Byte in VRAM
Because GPU memory buses are byte-addressable (minimum addressable unit is 8 bits), the quantizer packs $q_2$ into the high 4 bits and $q_1$ into the low 4 bits:

$$\text{Packed Byte} = (q_2 \ll 4) \ | \ q_1 = (\mathbf{1101}_2 \ll 4) \ | \ \mathbf{0101}_2 = \mathbf{11010101}_2 \quad (\text{Hex: } \mathbf{\text{0xD5}})$$

```
A Single 8-bit Byte in VRAM (0xD5):
┌────────────────────────┬────────────────────────┐
│  Bits 7..4: q2 = 13    │  Bits 3..0: q1 = 5     │
│       [1 1 0 1]        │       [0 1 0 1]        │
└────────────────────────┴────────────────────────┘
```

**Storage Result:** We compressed two weights from **4 bytes down to 1 byte** ($75\%$ memory bandwidth saved).

---

### Step 4: During Inference — The GPU Kernel Unpacks in Registers
During token generation, the GPU streams that single byte `0xD5` across the memory bus into a register. 

Inside the CUDA dequantization kernel (e.g. Marlin or AWQ kernel), the GPU unpacks them in **1 clock cycle**:

1. **Unpack $q_1$ (Low 4 bits)** using bitwise mask `0x0F` (`00001111`):
   $$\text{0xD5} \ \& \ \text{0x0F} = \mathbf{11010101}_2 \ \& \ \mathbf{00001111}_2 = \mathbf{00000101}_2 = \mathbf{5}$$
2. **Unpack $q_2$ (High 4 bits)** using bit-shift right `>> 4`:
   $$\text{0xD5} \gg 4 = \mathbf{00001101}_2 = \mathbf{13}$$
3. **Dequantize back to FP16 floating-point values**:
   Using the formula $\hat{W} = (q - Z) \times S$:
   * **Reconstructed $\hat{W}_1$**:
     $$\hat{W}_1 = (5 - 8) \times 0.04 = -3 \times 0.04 = \mathbf{-0.120}$$
     *(Original was `-0.125`; tiny rounding difference of `0.005`)*
   * **Reconstructed $\hat{W}_2$**:
     $$\hat{W}_2 = (13 - 8) \times 0.04 = +5 \times 0.04 = \mathbf{+0.200}$$
     *(Exact match!)*

Inside the GPU register, `-0.120` is formatted as a standard 16-bit IEEE floating-point binary string. The Tensor Core immediately multiplies `-0.120` with the incoming token's activation vector.

---

## 3. What Gets Saved vs. What Gets Discarded?

To resolve a common point of confusion:

### A. Group Scale ($S$) and Zero-Point ($Z$) ARE SAVED
To reconstruct the weights during inference, the GPU **must know $S$ and $Z$**. 
* **How are they stored?**
  In grouped quantization (`group_size = 128`), there is **one single Scale ($S$) and one Zero-point ($Z$) for every 128 weights**.
* **Memory Overhead Analysis**:
  * 128 weights in 4-bit = $128 \times 4\text{ bits} = 512\text{ bits} = \mathbf{64\text{ bytes}}$.
  * One scale $S$ in FP16 = $\mathbf{2\text{ bytes}}$.
  * One zero-point $Z$ in 4-bit = $\mathbf{0.5\text{ bytes}}$.
  * Overhead of storing $S$ and $Z$ = $\frac{2.5\text{ bytes}}{64\text{ bytes}} \approx \mathbf{3.9\%}$ extra VRAM!
* In any AWQ or GPTQ model repository (`model.safetensors`), you will find three distinct tensors per layer:
  1. `qweight`: The packed 4-bit weight integers.
  2. `scales`: The FP16 group scale tensor.
  3. `qzeros`: The packed integer zero-point tensor.

### B. What is NOT Saved:
1. **The GPTQ Hessian Matrix ($H = 2 X X^T$)**:
   The Hessian is only used **offline during the quantization process** to compute error-compensation adjustments for remaining unquantized weights. Once all weights are converted to INT4, the Hessian is deleted.
2. **The AWQ Activation Scaling Factor ($s$)**:
   In AWQ, the per-channel activation scaling factor $s$ is **mathematically fused into the preceding LayerNorm/RMSNorm weights** offline. At runtime, activations emerge naturally pre-scaled with zero runtime overhead.

---

## 4. Activation Quantization Across Deep Transformer Layers

Why can't we just quantize activations once at the input or embedding layer?

An LLM is not a simple embedding lookup. An LLM has **28 to 80 deep Transformer blocks**:

```
[Token ID: 9906] ──► [Embedding Layer] ──► [FP8 / FP16 Vectors]
                                                    │
┌────────────────── Transformer Layer i ────────────┴──────────────────────┐
│                                                                          │
│  1. RMSNorm:           X_norm = RMSNorm(X)                               │
│  2. Attention QKV:     Q = X_norm * W_q,  K = X_norm * W_k               │
│  3. Softmax:           Attn = Softmax(Q * K^T / sqrt(d)) * V             │
│  4. Residual Add (+):  X_mid = X + Attn   <-- Numbers accumulate & grow! │
│  5. RMSNorm:           X_mid_norm = RMSNorm(X_mid)                       │
│  6. SwiGLU / MLP:      MLP = (X_mid_norm * W_gate) * sigmoid(...) * W_up │
│  7. Residual Add (+):  X_out = X_mid + (MLP * W_down)                    │
│                                                                          │
└───────────────────────────────────────────────────┬──────────────────────┘
                                                    │
                        Intermediate numbers change dynamic range!
                        Must be re-quantized before Layer i+1!
```

Because additions, multiplications, and non-linearities (Softmax, SiLU) continuously alter the magnitude and scale of intermediate vectors, **Activation Quantization must occur dynamically at every single layer** across the entire network.
