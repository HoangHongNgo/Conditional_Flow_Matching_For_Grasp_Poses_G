import torch
import torch.nn as nn


class ScoringMLP(nn.Module):
    """Predict discrete grasp score logits from grasp pose and seed features."""

    def __init__(
        self,
        input_dim=133,
        num_classes=11,
        hidden_dim=256,
        dropout=0.1,
        grasp_dim=5,
        sample_embed_dim=None,
    ):
        """Initialize a GraspGen-style scoring classifier."""
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.grasp_dim = grasp_dim
        self.cond_dim = input_dim - grasp_dim
        self.sample_embed_dim = sample_embed_dim if sample_embed_dim is not None else hidden_dim

        if self.cond_dim <= 0:
            raise ValueError(
                f"input_dim must be larger than grasp_dim, got input_dim={input_dim}, "
                f"grasp_dim={grasp_dim}."
            )

        total_input_dim = self.sample_embed_dim + self.cond_dim
        head_hidden_dim = total_input_dim // 2
        head_bottleneck_dim = total_input_dim // 4

        self.grasp_norm = nn.LayerNorm(grasp_dim)
        self.condition_norm = nn.LayerNorm(self.cond_dim)
        self.sample_encoder = nn.Sequential(
            nn.Linear(grasp_dim, self.sample_embed_dim),
            nn.LayerNorm(self.sample_embed_dim),
            nn.ReLU(),
            nn.Linear(self.sample_embed_dim, self.sample_embed_dim),
            nn.LayerNorm(self.sample_embed_dim),
            nn.ReLU(),
        )

        head_layers = [
            nn.LayerNorm(total_input_dim),
            nn.Linear(total_input_dim, head_hidden_dim),
            nn.LayerNorm(head_hidden_dim),
            nn.ReLU(),
        ]
        if dropout > 0:
            head_layers.append(nn.Dropout(dropout))
        head_layers.extend(
            [
                nn.Linear(head_hidden_dim, head_bottleneck_dim),
                nn.LayerNorm(head_bottleneck_dim),
                nn.ReLU(),
            ]
        )
        if dropout > 0:
            head_layers.append(nn.Dropout(dropout))
        head_layers.append(nn.Linear(head_bottleneck_dim, num_classes))
        self.prediction_head = nn.Sequential(
            *head_layers
        )

    def forward(self, grasp_config):
        """Return class logits for grasp_config with shape [..., 133]."""
        if grasp_config.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dimension {self.input_dim}, got {grasp_config.shape[-1]}."
            )
        original_shape = grasp_config.shape[:-1]
        flat_config = grasp_config.reshape(-1, self.input_dim)  # [M, 133]
        grasp_pose = flat_config[:, :self.grasp_dim]  # [M, 5]
        seed_condition = flat_config[:, self.grasp_dim:]  # [M, 128]
        grasp_pose = self.grasp_norm(grasp_pose)  # [M, 5]
        seed_condition = self.condition_norm(seed_condition)  # [M, 128]
        sample_embedding = self.sample_encoder(grasp_pose)  # [M, sample_embed_dim]
        total_embedding = torch.cat([sample_embedding, seed_condition], dim=-1)
        flat_logits = self.prediction_head(total_embedding)  # [M, 11]
        return flat_logits.reshape(*original_shape, self.num_classes)


def logits_to_expected_score(logits):
    """Convert 11-class score logits into expected scores in [0, 1]."""
    num_classes = logits.shape[-1]
    score_bins = torch.arange(num_classes, device=logits.device, dtype=logits.dtype)
    score_bins = score_bins / max(num_classes - 1, 1)
    probabilities = torch.softmax(logits, dim=-1)
    return (probabilities * score_bins).sum(dim=-1)
