from dataclasses import dataclass

from hiad.constants import (
    SUPPORTED_TASK_TYPES,
    TASK_TYPE_THUMBNAIL,
)


@dataclass(frozen=True)
class PreparedInputRecord:
    task_name: str
    task_type: str
    image_path: str
    image_size: tuple[int, int]
    model_input_size: tuple[int, int]
    source_xywh: tuple[int, int, int, int] | None = None
    valid_source_hw: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if not self.task_name or self.task_type not in SUPPORTED_TASK_TYPES:
            raise ValueError("Prepared input task identity is invalid")
        if not self.image_path:
            raise ValueError("Prepared input image_path must be non-empty")
        for name, size in (
            ("image_size", self.image_size),
            ("model_input_size", self.model_input_size),
        ):
            if (
                not isinstance(size, tuple)
                or len(size) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                    for value in size
                )
            ):
                raise ValueError(f"{name} must be a positive integer pair")
        if self.task_type == TASK_TYPE_THUMBNAIL:
            if self.source_xywh is not None or self.valid_source_hw is not None:
                raise ValueError("Thumbnail records must not contain local source geometry")
            return
        if (
            not isinstance(self.source_xywh, tuple)
            or len(self.source_xywh) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in self.source_xywh
            )
        ):
            raise ValueError("Dynamic records require integer source_xywh")
        if (
            not isinstance(self.valid_source_hw, tuple)
            or len(self.valid_source_hw) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in self.valid_source_hw
            )
        ):
            raise ValueError("Dynamic records require positive valid_source_hw")
