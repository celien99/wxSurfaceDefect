from __future__ import annotations

from dataclasses import dataclass

_ODD_KERNEL_FIELDS = (
    "gaussian_kernel",
    "phase_smooth_kernel",
    "modulation_local_kernel",
    "modulation_std_kernel",
    "dark_background_kernel",
    "texture_local_kernel",
)


@dataclass
class FusionConfig:
    """程控条纹光 8→1 融合参数。"""

    gaussian_sigma: float = 0.5
    gaussian_kernel: int = 3
    input_max: float | None = None
    saturation_onset: float = 0.98
    x_phase_axis: int = 1
    y_phase_axis: int = 0
    phase_gradient_kernel: int = 3
    phase_smooth_kernel: int = 5
    modulation_confidence_floor: float = 0.08
    modulation_local_kernel: int = 31
    modulation_std_kernel: int = 15
    highpass_sigma: float = 3.0
    dark_background_kernel: int = 31
    texture_sigma_small: float = 2.0
    texture_sigma_large: float = 9.0
    texture_local_kernel: int = 15
    texture_period: int = 32
    texture_period_scales: tuple[float, ...] = (0.8, 1.0, 1.2)
    percentile_low: float = 2.0
    percentile_high: float = 98.0
    min_dynamic_range: float = 1e-3
    weight_phase: float = 0.35
    weight_scratch: float = 0.35
    weight_dark: float = 0.15
    weight_texture: float = 0.15
    output_blur_kernel: int = 3
    soft_threshold: float = 0.02

    def __post_init__(self) -> None:
        self._validate_kernels()
        self._validate_positive_scalars()
        self._validate_ranges()
        self._validate_axes()
        self._validate_weights()

    def _validate_kernels(self) -> None:
        if self.phase_gradient_kernel not in (1, 3, 5, 7):
            raise ValueError("phase_gradient_kernel must be 1, 3, 5, or 7")
        if self.output_blur_kernel < 1 or (
            self.output_blur_kernel > 1 and self.output_blur_kernel % 2 == 0
        ):
            raise ValueError("output_blur_kernel must be 1 or a positive odd integer")
        for name in _ODD_KERNEL_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer")

    def _validate_positive_scalars(self) -> None:
        if self.gaussian_sigma < 0:
            raise ValueError("gaussian_sigma must be non-negative")
        for name in (
            "highpass_sigma",
            "texture_sigma_small",
            "texture_sigma_large",
            "min_dynamic_range",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.texture_period <= 0:
            raise ValueError("texture_period must be positive")
        if not self.texture_period_scales or any(
            scale <= 0 for scale in self.texture_period_scales
        ):
            raise ValueError("texture_period_scales must contain positive values")
        if self.input_max is not None and self.input_max <= 0:
            raise ValueError("input_max must be positive when provided")

    def _validate_ranges(self) -> None:
        if not 0.0 <= self.percentile_low < self.percentile_high <= 100.0:
            raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
        if not 0.0 <= self.soft_threshold < 1.0:
            raise ValueError("soft_threshold must be in [0, 1)")
        if not 0.0 <= self.saturation_onset < 1.0:
            raise ValueError("saturation_onset must be in [0, 1)")
        if not 0.0 < self.modulation_confidence_floor <= 1.0:
            raise ValueError("modulation_confidence_floor must be in (0, 1]")

    def _validate_axes(self) -> None:
        if self.x_phase_axis not in (0, 1) or self.y_phase_axis not in (0, 1):
            raise ValueError("phase axes must be 0 or 1")
        if self.x_phase_axis == self.y_phase_axis:
            raise ValueError("x_phase_axis and y_phase_axis must be different")

    def _validate_weights(self) -> None:
        weights = (
            self.weight_phase,
            self.weight_scratch,
            self.weight_dark,
            self.weight_texture,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("fusion weights must be non-negative")
        if sum(weights) <= 0:
            raise ValueError("fusion weights must sum to a positive value")
