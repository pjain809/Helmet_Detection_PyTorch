
from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    test_file_path: str
    train_file_path: str
    valid_file_path: str
