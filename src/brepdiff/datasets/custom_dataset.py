from brepdiff.datasets.abc_dataset import ABCDataset


class CustomStepDataset(ABCDataset):
    """Dataset wrapper for custom STEP-derived UV-grid samples."""

    name = "custom_step"

    def get_data_list_path(self) -> str:
        return "./data/custom_uvgrid/custom_train.txt"

    def get_invalid_step_list_path(self) -> str:
        return "./data/custom_uvgrid/custom_invalid.txt" 
