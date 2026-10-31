"""Settings shared by every platform in the ReLU^2 / FP4 study, so the Mac (MLX), TPU (JAX) and B300 (PyTorch) runs train
the same model with the same optimizer on the same tokens. Mirrors anvil_r/mlx_gpt/train_mlx.py, where the Mac grid ran.

The MLP activation and the number-format recipe are the two factors under study; everything else is fixed here.
"""
from __future__ import annotations

CONFIGS = {
    # The Mac grid's model: 29M parameters (most in the 50,304 x 256 embedding and head), 16k tokens per step.
    "tiny": dict(d=256, layers=4, heads=4, seq=512, batch=32, vocab=50304),
    # GPT-2-small shape for the scale-up on B300 / TPU, where FP4 throughput and outlier growth start to matter.
    "gpt2s": dict(d=768, layers=12, heads=12, seq=1024, batch=32, vocab=50304),
}

# MLP activations. Gated ("packed") ones split c_fc into u | v with hidden He = round_up(8d/3, 64), so every activation
# has the same parameter count as ReLU^2's 4d MLP (within 3%).
ACTIVATIONS = {
    "relu2": "relu(x)^2  (modded-nanogpt; NVIDIA's NVFP4 12B run; Agrusa et al. 2026)",
    "lrelu2": "LeakyReLU(0.5)(x)^2  (every OpenAI parameter-golf record since 2026-03-23)",
    "swiglu": "silu(u) * v  (Llama-style; the FP4/FP8 outlier papers' activation)",
    "xielu": "xIELU, trainable per-layer alpha_p = alpha_n = 0.8 at init, beta = 0.5 (arXiv 2411.13010)",
    "s2glu_r2": "2*sigmoid(u) * relu(v)^2  (gated ReLU^2)",
    "eglu_r2n": "(e^tanh(u) - 1/e) / (e - 1/e) * relu(v)^2  (bounded EML gate in (0, 1))",
    "eml": "e^tanh(u) - ln(1 + relu(v))  (the 'EML dual-space' proposal; bounded, low crest factor)",
}
PACKED = {"swiglu", "s2glu_r2", "eglu_r2n", "eml"}

# Optimizer: Muon on the blocks' 2-D matrices, Adam (no bias correction, as mlx.optimizers.AdamW) on the rest.
MUON = dict(lr=0.02, momentum=0.95, nesterov=True, ns_coeffs=(3.4445, -4.7750, 2.0315), ns_steps=5)
ADAM = dict(lr=0.008, b1=0.9, b2=0.95, eps=1e-8)
SCHEDULE = dict(cooldown_frac=0.4, floor=0.1)   # constant, then linear decay over the last 40% to 0.1x
SOFTCAP = 15.0
HEAD_STD = 0.005
NORM_EPS = 1e-6
ROPE_BASE = 10000.0
XIELU = dict(alpha_p=0.8, alpha_n=0.8, beta=0.5, eps=-1e-6)
# The Mac stage-1 calibration found the default LR too low for 300-step runs (x1.25 gave ReLU^2 -45 mnat). Use the
# --lr-mult the stage-1 LR phase selects (see README); 1.0 reproduces the Mac grid exactly.
DEFAULT_LR_MULT = 1.0

DATA_REPO = "kjj0/fineweb10B-gpt2"            # modded-nanogpt's cached GPT-2-tokenized FineWeb
TRAIN_SHARD = "fineweb_train_000001.bin"      # contiguous chunks, step s uses tokens [s*B*T, (s+1)*B*T + 1)
VAL_SHARD = "fineweb_val_000000.bin"          # val = the first VAL_TOKENS tokens, batches of 16 sequences
VAL_TOKENS = 1 << 19


def hidden_for(act: str, d: int) -> tuple[int, int]:
    """(c_fc out features, c_proj in features)."""
    if act in PACKED:
        he = (8 * d // 3 + 63) // 64 * 64
        return 2 * he, he
    return 4 * d, 4 * d


def lr_at(step: int, n: int, base: float) -> float:
    cd = int(n * (1 - SCHEDULE["cooldown_frac"]))
    t = min(max((step - cd) / max(1, n - cd), 0.0), 1.0)
    return base * (1.0 - (1.0 - SCHEDULE["floor"]) * t)


# Number-format recipes. "sim_*" = fake quantization with the same numerics on any device (ports of
# anvil_r/mlx_gpt/lowp.py); the others use real low-precision hardware paths.
SIM_RECIPES = {   # (x / w format, gradient format, stochastic rounding on gradients, Hadamard-16 on the wgrad GEMM)
    "sim_fp8": ("e4m3", "e5m2", False, False),
    "sim_int8": ("int8", "int8", True, False),
    "sim_mxfp4": ("mxfp4", "mxfp4", True, False),
    "sim_nvfp4": ("nvfp4", "nvfp4", True, False),
    "sim_nvfp4h": ("nvfp4", "nvfp4", True, True),
}
