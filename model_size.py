import torch
from models.economicgrasp import economicgrasp, ecograsp


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_parameters(model):
    total = 0
    for name, param in model.named_parameters():
        num = param.numel()
        print(f"{name}: {num}")
        total += num
    print(f"\nTotal Parameters: {total}")


def model_size_mb(model):
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 4 / (1024**2)  # float32 = 4 bytes
    return size_mb


model = economicgrasp(seed_feat_dim=512, is_training=True)

print("Total params:", count_parameters(model))
print("Trainable params:", count_trainable_parameters(model))
print("Model size (MB):", model_size_mb(model))

# print_model_parameters(model)
