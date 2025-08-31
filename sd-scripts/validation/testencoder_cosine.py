import torch
from transformers import CLIPTokenizer, CLIPTextModel

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_te(model_id):
    tok = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    te  = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    te = te.to(device).eval()
    return tok, te

tok_base, te_base = load_te("runwayml/stable-diffusion-v1-5")
tok_ft,   te_ft   = load_te("../tools/output_merged_sd")  # your merged model path

def embed_sentence(tok, te, text):
    with torch.no_grad():
        ids = tok([text], padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        out = te(**ids).last_hidden_state  # [1, 77, 768]
        # Use mean-pooled text embedding (common trick for inspection)
        return out.mean(dim=1).squeeze(0)  # [768]

probe = "an energetic instrumental track with heavy use of synthesizers, the track would be used in a video game setting mel spectrogram"

e_base = embed_sentence(tok_base, te_base, probe)
e_ft   = embed_sentence(tok_ft,   te_ft,   probe)

cos = torch.nn.functional.cosine_similarity(e_base, e_ft, dim=0).item()
l2  = torch.linalg.vector_norm(e_ft - e_base).item()
print("cosine(base,ft) =", cos, "  L2 shift =", l2)

probe = "instrumental track"

e_base = embed_sentence(tok_base, te_base, probe)
e_ft   = embed_sentence(tok_ft,   te_ft,   probe)

cos = torch.nn.functional.cosine_similarity(e_base, e_ft, dim=0).item()
l2  = torch.linalg.vector_norm(e_ft - e_base).item()
print("cosine(base,ft) =", cos, "  L2 shift =", l2)
