#!/usr/bin/env python3
"""
Convert Hugging Face Gemma 4 Text (E2B) weights to Espresso format.

Requires:
    pip install torch transformers

Usage:
    ./scripts/convert_weights_gemma4.py --model google/gemma-4-E2B-it --output /tmp/gemma4-e2b
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

# Add the project root to sys.path so we can import from gemma4/
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def save_blob(path: Path, tensor: torch.Tensor):
    """Saves a tensor in Espresso BLOBFILE format (128-byte header + float16 payload)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # 128-byte zero header
    header = bytearray(128)
    
    # Convert to float16 and numpy
    # Ensure it's in the right shape and on CPU
    data = tensor.detach().cpu().to(torch.float16).numpy().tobytes()
    
    with open(path, "wb") as f:
        f.write(header)
        f.write(data)

def convert_gemma4_weights(model_id: str, output_dir: Path):
    # Try to import from the local gemma4 directory first
    try:
        print("Using transformers Gemma4 implementation.")
        from transformers import AutoConfig as Gemma4Config
        from transformers import AutoModelForCausalLM as Gemma4ForConditionalGeneration
    except ImportError:
        print("Gemma4 is not Found")
    

    print(f"Loading model {model_id}...")
    # For Gemma 4 E2B, we might need trust_remote_code if not using local files
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="cpu",
        trust_remote_code=True
    )
    config = model.config
    
    # Extract text config (Gemma 4 is multimodal, we want the text part)
    if hasattr(config, "text_config") and config.text_config is not None:
        text_config = config.text_config
    else:
        text_config = config

    state_dict = model.state_dict()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = {
        "name": "gemma4",
        "nLayer": text_config.num_hidden_layers,
        "nHead": text_config.num_attention_heads,
        "nKVHead": text_config.num_key_value_heads,
        "dModel": text_config.hidden_size,
        "headDim": text_config.head_dim,
        "hiddenDim": text_config.intermediate_size,
        "vocab": text_config.vocab_size,
        "maxSeq": text_config.max_position_embeddings,
        "normEps": text_config.rms_norm_eps,
        "slidingWindow": getattr(text_config, "sliding_window", 512),
        "vocabSizePerLayerInput": getattr(text_config, "vocab_size_per_layer_input", 0),
        "hiddenSizePerLayerInput": getattr(text_config, "hidden_size_per_layer_input", 0),
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Converting weights...")
    
    # 1. Embeddings
    print("  Embeddings...")
    # Main token embeddings
    if "model.embed_tokens.weight" in state_dict:
        save_blob(output_dir / "embeddings" / "token.bin", state_dict["model.embed_tokens.weight"])
    
    # Per-layer input embeddings (specific to Gemma 4 E2B)
    # The key might vary depending on the implementation, but let's check for likely candidates
    per_layer_emb_keys = [
        "model.language_model.embed_tokens_per_layer.weight",
        "model.embed_tokens_per_layer.weight",
        "language_model.embed_tokens_per_layer.weight"
    ]
    for key in per_layer_emb_keys:
        if key in state_dict:
            print(f"    Found per-layer embeddings: {key}")
            save_blob(output_dir / "embeddings" / "token_per_layer.bin", state_dict[key])
            break

    # 2. Final Norm & Head
    print("  Final layers...")
    norm_keys = ["model.norm.weight", "model.language_model.norm.weight"]
    for key in norm_keys:
        if key in state_dict:
            save_blob(output_dir / "final_norm.bin", state_dict[key])
            break
            
    head_keys = ["lm_head.weight"]
    for key in head_keys:
        if key in state_dict:
            save_blob(output_dir / "lm_head.bin", state_dict[key])
            break

    # 3. Layers
    print("  Layers...")
    for i in tqdm(range(text_config.num_hidden_layers)):
        # Handle potential prefixing (model.layers.i or model.language_model.layers.i)
        layer_prefix = None
        for p in [f"model.layers.{i}", f"model.language_model.layers.{i}"]:
            if f"{p}.self_attn.q_proj.weight" in state_dict:
                layer_prefix = p
                break
        
        if layer_prefix is None:
            print(f"Warning: Could not find weights for layer {i}")
            continue

        layer_dir = output_dir / "layers" / str(i)
        
        # Attention
        save_blob(layer_dir / "wq.bin", state_dict[f"{layer_prefix}.self_attn.q_proj.weight"])
        save_blob(layer_dir / "wk.bin", state_dict[f"{layer_prefix}.self_attn.k_proj.weight"])
        save_blob(layer_dir / "wv.bin", state_dict[f"{layer_prefix}.self_attn.v_proj.weight"])
        save_blob(layer_dir / "wo.bin", state_dict[f"{layer_prefix}.self_attn.o_proj.weight"])
        
        # Norms
        save_blob(layer_dir / "rms_att.bin", state_dict[f"{layer_prefix}.input_layernorm.weight"])
        save_blob(layer_dir / "rms_ffn.bin", state_dict[f"{layer_prefix}.post_attention_layernorm.weight"])
        
        # MLP
        save_blob(layer_dir / "w1.bin", state_dict[f"{layer_prefix}.mlp.gate_proj.weight"])
        save_blob(layer_dir / "w2.bin", state_dict[f"{layer_prefix}.mlp.down_proj.weight"])
        save_blob(layer_dir / "w3.bin", state_dict[f"{layer_prefix}.mlp.up_proj.weight"])
        
        # Q/K Norms
        if f"{layer_prefix}.self_attn.q_norm.weight" in state_dict:
            save_blob(layer_dir / "q_norm.bin", state_dict[f"{layer_prefix}.self_attn.q_norm.weight"])
        if f"{layer_prefix}.self_attn.k_norm.weight" in state_dict:
            save_blob(layer_dir / "k_norm.bin", state_dict[f"{layer_prefix}.self_attn.k_norm.weight"])

        # Per-layer input projection (specific to Gemma 4 E2B)
        # Each layer might have its own projection from the per-layer embedding
        if f"{layer_prefix}.per_layer_input_proj.weight" in state_dict:
            save_blob(layer_dir / "per_layer_input_proj.bin", state_dict[f"{layer_prefix}.per_layer_input_proj.weight"])

    print(f"Done! Weights saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="google/gemma-4-e2b-it")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    convert_gemma4_weights(args.model_id, Path(args.output_dir))
