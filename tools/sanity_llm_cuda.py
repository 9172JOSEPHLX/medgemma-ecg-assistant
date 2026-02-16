import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    import os
    assert os.getenv('HF_TOKEN'), 'HF_TOKEN missing (set it before running).'
    print("torch:", torch.__version__, "cuda:", torch.version.cuda, "avail:", torch.cuda.is_available())
    assert torch.cuda.is_available(), "CUDA not available"

    model_id = "sshleifer/tiny-gpt2"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")

    prompt = "QC status: PASS. Next steps:"
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=20)
    print(tok.decode(out[0], skip_special_tokens=True))
    print("device:", next(model.parameters()).device)

if __name__ == "__main__":
    main()
