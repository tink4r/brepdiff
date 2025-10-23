import os

from brepdiff.datasets.abc_dataset import ABCDataset


class CustomStepDataset(ABCDataset):
    """Dataset wrapper for custom STEP-derived UV-grid samples."""

    name = "custom_step"

    def get_data_list_path(self) -> str:
        base_dir = os.path.dirname(self.config.h5_path)
        data_list_path = os.path.join(base_dir, f"custom_{self.split}.txt")
        return data_list_path

    def get_invalid_step_list_path(self) -> str:
        base_dir = os.path.dirname(self.config.h5_path)
        invalid_path = os.path.join(base_dir, "custom_invalid.txt")
        return invalid_path
