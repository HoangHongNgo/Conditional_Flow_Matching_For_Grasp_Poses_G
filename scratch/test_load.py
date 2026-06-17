import os
import sys
# Adding project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from models.economicgrasp import liteptgrasp

def test_loading(checkpoint_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # Init the model
    net = liteptgrasp(seed_feat_dim=512, is_training=False)
    net.to(device)
    model_dict = net.state_dict()
    
    if not os.path.isfile(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    print(f"=> Loading checkpoint from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"Critical Error: Could not load checkpoint file: {e}")
        return

    # --- 1. Raw Key Discovery ---
    print("\n--- Raw Checkpoint Keys ---")
    raw_keys = list(checkpoint.keys())
    for k in raw_keys:
        val_desc = type(checkpoint[k]).__name__
        if isinstance(checkpoint[k], torch.Tensor):
            val_desc += f" {checkpoint[k].size()}"
        elif isinstance(checkpoint[k], dict):
            val_desc += f" (Sub-keys: {list(checkpoint[k].keys())[:5]}...)"
        print(f"  - {k} : {val_desc}")

    # --- 2. Weight Extraction Logic ---
    if 'model_state_dict' in checkpoint:
        pretrained_dict = checkpoint['model_state_dict']
        print("\nUsing 'model_state_dict' for weights.")
    elif 'state_dict' in checkpoint:
        pretrained_dict = checkpoint['state_dict']
        print("\nUsing 'state_dict' for weights.")
    else:
        pretrained_dict = checkpoint
        print("\nUsing root dictionary for weights.")

    # --- 3. Key Normalization & Mapping ---
    # Many checkpoints use different prefixes. Let's analyze prefixes in checkpoint
    ckpt_prefixes = set()
    for k in pretrained_dict.keys():
        ckpt_prefixes.add(k.split('.')[0])
    print(f"\nDetected prefixes in checkpoint: {ckpt_prefixes}")

    model_prefixes = set()
    for k in model_dict.keys():
        model_prefixes.add(k.split('.')[0])
    print(f"Expected prefixes in model: {model_prefixes}")

    # Remap keys (handling 'module.' and redundant 'backbone.' prefixes)
    new_pretrained_dict = {}
    for k, v in pretrained_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.'
        
        # If it was saved with 'backbone.' prefix already
        if name.startswith('backbone.'):
            new_key = name
        else:
            new_key = 'backbone.' + name
            
        # Handle double backbone backbone
        if new_key.startswith('backbone.backbone.'):
            new_key = new_key.replace('backbone.backbone.', 'backbone.')
            
        new_pretrained_dict[new_key] = v

    # --- 4. Filtering and Comparison ---
    matched_dict = {}
    mismatched_size = []
    missing_in_ckpt = []
    
    for k, v in model_dict.items():
        if k in new_pretrained_dict:
            if v.size() == new_pretrained_dict[k].size():
                matched_dict[k] = new_pretrained_dict[k]
            else:
                mismatched_size.append(k)
        else:
            missing_in_ckpt.append(k)
            
    unexpected_keys = [k for k in new_pretrained_dict.keys() if k not in model_dict]

    # --- 5. Loading ---
    net.load_state_dict(matched_dict, strict=False)
    
    print("\n--- Loading Results ---")
    print(f"Total model parameters keys: {len(model_dict)}")
    print(f"Successfully matched & loaded: {len(matched_dict)}")
    
    if len(mismatched_size) > 0:
        print(f"\nMismatched size (skipped) [{len(mismatched_size)}]:")
        for k in mismatched_size[:5]:
            print(f"  - {k} (Model: {model_dict[k].size()}, Ckpt: {new_pretrained_dict[k].size()})")

    # --- 6. Detailed Analysis of Missing Keys ---
    if len(missing_in_ckpt) > 0:
        print(f"\n--- MISSING KEYS ANALYSIS [{len(missing_in_ckpt)}] ---")
        backbone_missing = [k for k in missing_in_ckpt if k.startswith('backbone.')]
        other_missing = [k for k in missing_in_ckpt if not k.startswith('backbone.')]
        
        print(f"Missing in Backbone ({len(backbone_missing)} keys):")
        for k in backbone_missing[:20]:
            print(f"  - {k}")
        if len(backbone_missing) > 20: print(f"  ... and {len(backbone_missing)-20} more.")
        
        print(f"\nMissing in other modules ({len(other_missing)} keys):")
        modules = {}
        for k in other_missing:
            mod = k.split('.')[0]
            modules[mod] = modules.get(mod, 0) + 1
        for mod, count in modules.items():
            print(f"  - {mod}.* : {count} keys")

    # --- 7. "WHY" Comparison Section ---
    print("\n--- COMPARISON ANALYSIS ---")
    if len(matched_dict) == 0:
        print("CRITICAL: 0 keys matched. This usually means:")
        print("1. The weights are nested deeper in the checkpoint (e.g. checkpoint['state_dict']).")
        print("2. The key prefixes are completely different (checkpoint has 'module.backbone...' or similar).")
        print(f"First 5 checkpoint keys: {list(new_pretrained_dict.keys())[:5]}")
        print(f"First 5 model keys: {list(model_dict.keys())[:5]}")
    else:
        print(f"Loaded {len(matched_dict)} keys. Analysis of remaining {len(missing_in_ckpt)} missing:")
        if len(backbone_missing) > 0:
            print("- Backbone keys missing: Check if the LitePT version in the model matches the checkpoint version.")
            print("- Checkpoint might be a partial model (e.g. only encoder, no decoder).")

    print("\n=> LOAD TEST FINISHED.")

if __name__ == '__main__':
    target_ckpt = 'checkpoints/model_best.pth'
    test_loading(target_ckpt)
