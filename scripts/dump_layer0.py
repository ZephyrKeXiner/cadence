from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "Models" / "Qwen3-4B"
DUMP_DIR = ROOT / "Models" / "dumps"

if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_DIR),
    torch_dtype=torch.float32,
    local_files_only=True,
)
model.eval()

input_ids = torch.tensor([[9707, 11, 1879, 0]])

hooks = {}


def save(name):
    def hook(module, input, output):
        hooks[name] = output[0] if isinstance(output, tuple) else output

    return hook


model.model.embed_tokens.register_forward_hook(save("embed"))
model.model.layers[0].input_layernorm.register_forward_hook(save("input_ln"))
model.model.layers[0].self_attn.register_forward_hook(save("attn"))
model.model.layers[0].post_attention_layernorm.register_forward_hook(save("post_ln"))
model.model.layers[0].mlp.register_forward_hook(save("mlp"))
model.model.layers[0].register_forward_hook(save("layer0_out"))

with torch.no_grad():
    model(input_ids)

DUMP_DIR.mkdir(parents=True, exist_ok=True)
for name, t in hooks.items():
    t = t.detach().to(torch.float32).cpu().numpy()
    print(name, t.shape, t.flatten()[:5])
    t.tofile(DUMP_DIR / f"layer0_{name}.bin")
