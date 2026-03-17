"""
Image-to-Image Schrödinger Bridge (I2SB) Engine
================================================

Implements the simulation-free I2SB formulation for bridging between degraded
(low-light) and clean image distributions.

Mathematical foundation:
    Forward transport:  q(x_t | x_0, x_1) = N(x_t; (1-t)*x_0 + t*x_1, sigma_t^2 * I)
    Bridge matching:    L = || v_theta(x_t, t) - (x_1 - x_0) ||^2
    Reverse sampling:   x_{t+dt} = x_t + v_theta(x_t, t) * dt

Here x_0 is the degraded (low-light) image and x_1 is the clean target.
The bridge variance sigma_t^2 = sigma_max^2 * t * (1 - t) ensures zero
variance at the boundary conditions t=0 (x_0) and t=1 (x_1).

Reference: Liu et al., "I2SB: Image-to-Image Schrödinger Bridge", ICML 2023.
"""

import math
import torch
import torch.nn as nn


class NoiseSchedule:
    """
    Bridge variance schedule for I2SB.

    The variance sigma_t^2 = sigma_max^2 * t * (1 - t) is a parabolic schedule
    that is zero at the boundary conditions (t=0 and t=1) and maximal at t=0.5.
    This is the natural choice for a Schrödinger Bridge connecting two known
    endpoints: it introduces the most stochasticity in the middle of the
    transport path where uncertainty about the true mapping is highest.

    Args:
        sigma_max: Maximum standard deviation scale factor. Controls the
            overall noise level in the bridge. Larger values allow more
            exploration but require more NFE steps for accurate sampling.
    """

    def __init__(self, sigma_max=0.5):
        self.sigma_max = sigma_max

    def sigma_t(self, t):
        """
        Compute the standard deviation at time t.

        Args:
            t: Time value(s) in [0, 1]. Can be a scalar or tensor.

        Returns:
            sigma_t: Standard deviation(s) at time t.
                     Shape matches input t.
        """
        # sigma_t = sigma_max * sqrt(t * (1 - t))
        # The sqrt ensures we work with standard deviations, not variances.
        return self.sigma_max * torch.sqrt(t * (1.0 - t) + 1e-8)

    def sigma_t_squared(self, t):
        """
        Compute the variance at time t.

        Args:
            t: Time value(s) in [0, 1].

        Returns:
            sigma_t^2: Variance(s) at time t.
        """
        return (self.sigma_max ** 2) * t * (1.0 - t)


class ISBEngine:
    """
    Simulation-free Image-to-Image Schrödinger Bridge engine.

    This class encapsulates the forward transport (sampling x_t given endpoints)
    and the reverse iterative sampling procedure. It does NOT contain learnable
    parameters; those live in the velocity network (ISBDenoiser).

    The "simulation-free" aspect means we never solve the SDE during training.
    Instead, we:
      1. Sample t ~ U(0, 1)
      2. Compute x_t = (1-t)*x_0 + t*x_1 + sigma_t * eps  (forward transport)
      3. Train v_theta to predict the velocity (x_1 - x_0) from (x_t, t)

    At inference, we integrate the learned velocity field from t=0 to t=1 using
    simple Euler steps (a predictor-only approach that is sufficient when NFE
    is adequately large; a corrector step is omitted to not halve the effective
    NFE on the P40 where compute is precious).

    Args:
        noise_schedule: A NoiseSchedule instance.
        nfe: Number of Function Evaluations for reverse sampling. Higher values
            give better quality but are linearly slower. Default 20 is a
            pragmatic choice for the P40 (no Tensor Cores, ~12 TFLOPS FP32).
    """

    def __init__(self, noise_schedule, nfe=20, reverse_noise_scale=0.5):
        self.noise_schedule = noise_schedule
        self.nfe = nfe
        self.reverse_noise_scale = float(reverse_noise_scale)
        if self.reverse_noise_scale < 0.0:
            raise ValueError(
                f"ISBEngine: reverse_noise_scale={self.reverse_noise_scale} is invalid. "
                "Expected a value >= 0."
            )

    def q_sample(self, x_0, x_1, t):
        """
        Sample from the forward bridge transport q(x_t | x_0, x_1).

        Implements:  x_t = (1 - t) * x_0 + t * x_1 + sigma_t * eps
        where eps ~ N(0, I).

        This is the core training-time operation: given a paired sample
        (x_0=low-light, x_1=clean), we produce a noisy intermediate x_t
        at a random time t for the velocity network to denoise.

        Args:
            x_0: Degraded images. Shape [b, c, h, w].
            x_1: Clean target images. Shape [b, c, h, w].
            t: Time values. Shape [b] with values in [0, 1].

        Returns:
            x_t: Noisy intermediate samples. Shape [b, c, h, w].
        """
        # Reshape t for broadcasting: [b] -> [b, 1, 1, 1]
        t_expanded = t[:, None, None, None]

        # Deterministic interpolation (the mean of the Gaussian)
        mean = (1.0 - t_expanded) * x_0 + t_expanded * x_1

        # Stochastic perturbation
        sigma = self.noise_schedule.sigma_t(t_expanded)
        eps = torch.randn_like(x_0)

        x_t = mean + sigma * eps
        return x_t

    @staticmethod
    def compute_velocity_target(x_0, x_1):
        """
        Compute the bridge matching velocity target.

        In the I2SB formulation, the optimal velocity field that transports
        the degraded distribution to the clean distribution is simply the
        displacement vector (x_1 - x_0). This is a consequence of the linear
        interpolation path chosen for the bridge.

        The velocity network v_theta is trained to predict this target,
        and the bridge matching loss is:
            L = || v_theta(x_t, t) - (x_1 - x_0) ||^2

        Args:
            x_0: Degraded images. Shape [b, c, h, w].
            x_1: Clean target images. Shape [b, c, h, w].

        Returns:
            velocity_target: Displacement vectors. Shape [b, c, h, w].
        """
        return x_1 - x_0

    def reverse_sample(self, velocity_network_fn, x_0, cond, nfe=None):
        """
        Iterative reverse sampling from x_0 (degraded) toward x_1 (clean).

        Uses Euler integration of the learned velocity field:
            x_{t+dt} = x_t + v_theta(x_t, cond, t) * dt

        This is a "predictor-only" approach. A Langevin corrector step could
        improve quality but would halve the effective NFE. On the P40,
        maximizing the number of meaningful velocity evaluations per unit
        compute is more important than correction refinement.

        The integration proceeds from t=0 (x_0, the degraded image) to
        t=1 (x_1, the clean target). Each step takes a fixed-size Euler
        step along the predicted velocity direction.

        Args:
            velocity_network_fn: Callable (x_t, cond, t_batch) -> v_pred.
                This is typically ISBDenoiser.forward or a wrapper thereof.
            x_0: Degraded input images. Shape [b, c, h, w].
            cond: Conditioning information (illumination features).
                Shape depends on the velocity network's expectation.
            nfe: Override for number of function evaluations.
                If None, uses self.nfe.

        Returns:
            x_t: Enhanced images after NFE steps. Shape [b, c, h, w].
            intermediates: List of (t_value, mse_to_x0) tuples for monitoring.
                MSE is computed against x_0 to track how far the state has
                moved from the degraded input.
        """
        if nfe is None:
            nfe = self.nfe

        b = x_0.shape[0]
        device = x_0.device
        dtype = x_0.dtype

        # Time steps from t=0 (degraded) to t=1 (clean)
        # We use nfe+1 points to get nfe intervals
        ts = torch.linspace(0.0, 1.0, nfe + 1, device=device, dtype=dtype)
        dt = 1.0 / nfe

        x_t = x_0.clone()
        intermediates = []

        for i in range(nfe):
            t_val = ts[i]
            t_batch = torch.full((b,), t_val, device=device, dtype=dtype)

            # Predict velocity at current state
            v_pred = velocity_network_fn(x_t, cond, t_batch)

            # Euler step: x_{t+dt} = x_t + v_theta * dt
            x_t = x_t + v_pred * dt

            # Track progress: compute MSE to x_0 to show the state is moving
            with torch.no_grad():
                mse_from_x0 = ((x_t - x_0) ** 2).mean().item()
                intermediates.append((t_val.item(), mse_from_x0))

        return x_t, intermediates

    def reverse_sample_fast(self, velocity_network_fn, x_0, cond, nfe=None,
                             predict_x0=False):
        """
        Fast reverse sampling with optional x0-prediction mode.

        Two modes:
        - predict_x0=False (velocity mode): x_{t+dt} = x_t + v_pred * dt
        - predict_x0=True (x0-prediction mode): network outputs predicted_x0,
          then x_{t-dt} is computed via DDPM-style posterior interpolation.

        Intermediate results are clamped to [0, 1] (requirement #6).

        Args:
            velocity_network_fn: Callable (x_t, cond, t_batch) -> prediction.
            x_0: Starting point. Shape [b, c, h, w].
            cond: Conditioning information (e.g., x1 for SB).
            nfe: Override for number of function evaluations.
            predict_x0: If True, network predicts x0 directly.

        Returns:
            x_t: Enhanced images. Shape [b, c, h, w].
        """
        if nfe is None:
            nfe = self.nfe

        b = x_0.shape[0]
        device = x_0.device
        dtype = x_0.dtype

        if predict_x0:
            # x0-prediction mode: start from x1 (cond), step toward x0 (clean)
            # Time goes from t=1 (corrupted) to t=0 (clean)
            x_t = x_0.clone()  # start from x1
            dt = 1.0 / nfe

            for i in range(nfe):
                t_val = 1.0 - i * dt  # t: 1.0 -> dt
                t_next = max(t_val - dt, 0.0)
                t_batch = torch.full((b,), t_val, device=device, dtype=dtype)

                # Network predicts clean x0 from current x_t
                predicted_x0 = velocity_network_fn(x_t, cond, t_batch)
                predicted_x0 = predicted_x0.clamp(0.0, 1.0)

                if t_next <= 0:
                    # Last step: output is the prediction
                    x_t = predicted_x0
                else:
                    # Interpolate toward predicted_x0
                    # x_{t-dt} = (t_next/t_val) * x_t + (1 - t_next/t_val) * predicted_x0
                    # This moves x_t closer to predicted_x0 proportionally
                    ratio = t_next / t_val
                    x_t = ratio * x_t + (1.0 - ratio) * predicted_x0

                    # Add small noise for stochasticity (optional, annealed)
                    sigma_t = self.noise_schedule.sigma_t(
                        torch.tensor(t_next, device=device, dtype=dtype)
                    )
                    if sigma_t > 1e-6 and self.reverse_noise_scale > 0.0:
                        noise_scale = sigma_t * self.reverse_noise_scale
                        x_t = x_t + noise_scale * torch.randn_like(x_t)

                x_t = x_t.clamp(0.0, 1.0)  # intermediate clamp

            return x_t
        else:
            # Velocity mode (original): x_{t+dt} = x_t + v * dt
            dt = 1.0 / nfe
            x_t = x_0.clone()
            for i in range(nfe):
                t_val = i * dt
                t_batch = torch.full((b,), t_val, device=device, dtype=dtype)
                v_pred = velocity_network_fn(x_t, cond, t_batch)
                x_t = x_t + v_pred * dt
                x_t = x_t.clamp(0.0, 1.0)  # intermediate clamp

            return x_t


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for continuous time values.

    Maps scalar time t in [0, 1] to a high-dimensional embedding using
    sinusoidal functions at logarithmically-spaced frequencies. This is
    the standard approach from "Attention Is All You Need" adapted for
    continuous diffusion time.

    The embedding dimension should be large enough to encode fine-grained
    time information. A dimension of 4*n_feat is typical.

    Args:
        dim: Output embedding dimension.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: Time values. Shape [b].

        Returns:
            emb: Time embeddings. Shape [b, dim].
        """
        half_dim = self.dim // 2
        scale = math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=t.dtype) * -scale
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 != 0:
            # Handle odd dimensions by zero-padding
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb
