import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from typing import Optional, Tuple
from accelerate import Accelerator

torch.autograd.set_detect_anomaly(True)


class AutoregressiveMlpBlock(nn.Module):
    def __init__(self, mlp_dim: int, in_features: int) -> None:
        super().__init__()
        self.dense1 = nn.Linear(in_features, mlp_dim)
        self.dense2 = nn.Linear(mlp_dim, in_features)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dense1(x)
        y = self.gelu(y)
        return self.dense2(y)


class AutoregressiveMixerBlock(nn.Module):
    def __init__(
        self,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
        num_tokens: int,
        hidden_dim: int,
        batch_size: int,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.token_mixing = AutoregressiveMlpBlock(tokens_mlp_dim, num_tokens)
        self.channel_mixing = AutoregressiveMlpBlock(channels_mlp_dim, hidden_dim)
        self.register_buffer(
            "causal_mask", torch.tril(torch.ones(num_tokens, num_tokens))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layer_norm1(x)
        y = y.transpose(1, 2)

        # Apply causal mask
        y = y.unsqueeze(-1)
        mask = self.causal_mask.unsqueeze(0).unsqueeze(
            0
        )  # (1, 1, num_tokens, num_tokens)
        mask = mask.expand(
            self.batch_size, self.hidden_dim, self.num_tokens, self.num_tokens
        )  # (batch_size, hidden_dim, num_tokens, num_tokens)
        y = y * mask  # (batch_size, hidden_dim, num_tokens, num_tokens)
        y = y.sum(dim=2)  # (batch_size, hidden_dim, num_tokens)

        # Token mixing
        y = self.token_mixing(y)
        y = y.transpose(1, 2)
        # x = x + y

        # Channel mixing
        y = self.layer_norm2(y)
        # output = x + self.channel_mixing(y)
        output = self.channel_mixing(y)

        return output


class AutoregressiveMlpMixer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        batch_size: int,
        num_blocks: int,
        hidden_dim: int,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
        *,
        initial_lr: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.98),
        epsilon: float = 1e-8,
        weight_decay: float = 1e-5,
        system_height=4,
        system_width=4,
        patch_size=2,
        input_channel: int = 2,  # Changed to 2 for one-hot encoding (0 and 1)
        model_name: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.input_channel = input_channel
        self.num_tokens = (system_height * system_width) // (patch_size**2)

        self.stem = nn.Conv2d(
            input_channel, hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        self.mixer_blocks = nn.ModuleList(
            [
                AutoregressiveMixerBlock(
                    tokens_mlp_dim,
                    channels_mlp_dim,
                    self.num_tokens,
                    hidden_dim,
                    batch_size,
                )
                for _ in range(num_blocks)
            ]
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

        self.optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=initial_lr,
            betas=betas,
            eps=epsilon,
            weight_decay=weight_decay
        )
        self.to(device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        
        x = self.stem(inputs)
        x = rearrange(x, "b c h w -> b (h w) c")

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)
        x = x.mean(dim=1)

        x = self.head(x)

        return x

    @torch.jit.export
    def sample(self):
        device = next(self.parameters()).device
        x = torch.zeros(
            (self.batch_size, self.input_channel, self.num_tokens, self.num_tokens),
            device=device,
        )

        for i in range(self.num_tokens * self.num_tokens):
            probs = F.softmax(self.forward(x), dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)
            x = x.view(self.batch_size, self.input_channel, -1)
            x[:, 0, i] = 1 - sampled
            x[:, 1, i] = sampled
            x = x.view(
                self.batch_size, self.input_channel, self.num_tokens, self.num_tokens
            )

        return x

    @torch.jit.export
    def log_probability(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = inputs.size()

        # Prepare the shifted input data
        data = torch.zeros_like(inputs)
        data[:, :, 1:, :] = inputs[:, :, :-1, :]

        # Forward pass through the model
        logits = self.forward(data)

        # Reshape logits and inputs for element-wise multiplication
        logits = logits.view(batch_size, -1, 2)
        inputs_flat = inputs.view(batch_size, 2, height * width).permute(0, 2, 1)

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)

        # Element-wise multiplication and sum
        total = torch.sum(inputs_flat * probs, dim=-1)

        # Compute log probabilities
        logp = torch.sum(torch.log(total), dim=-1)

        return logp
    

    @torch.jit.export
    def local_energy(self, inputs: torch.Tensor, Omega: float, delta: float, V: torch.Tensor) -> torch.Tensor:
        device = inputs.device
        batch_size, _, height, width = inputs.size()
        N = height * width
        
        # Rabi term: -Ω/2 Σ_i (σ_i^x)
        rabi_term = torch.zeros(batch_size, device=device)
        for i in range(N):
            flipped_inputs = inputs.clone()
            flipped_inputs[:, :, i // width, i % width] = flipped_inputs[:, [1, 0], i // width, i % width]
            log_psi = self.log_probability(inputs)
            log_psi_flipped = self.log_probability(flipped_inputs)
            rabi_term -= (Omega / 2) * torch.exp(log_psi_flipped - log_psi)
        
        # Detuning term: -Δ Σ_i n_i
        detuning_term = -delta * inputs[:, 1].sum(dim=[1, 2])
        
        # Interaction term: Σ_{i<j} V_ij n_i n_j
        inputs_flat = inputs[:, 1].view(batch_size, -1)
        interaction_term = torch.einsum('bi,ij,bj->b', inputs_flat, V, inputs_flat)
        
        # Total local energy
        local_energy = rabi_term + detuning_term + interaction_term
        
        return local_energy

    @torch.jit.export
    def update_parameters(
        self,
        logp: torch.Tensor,
        energies: torch.Tensor,
        mean_energy: float,
        accelerator: Accelerator
    ):
        self.optimizer.zero_grad()
        loss = (logp * (energies - mean_energy)).mean()

        # Back propagate
        accelerator.backward(loss)

        # Clip gradients (optional, but often helpful)
        accelerator.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # Update parameters
        self.optimizer.step()

        return
