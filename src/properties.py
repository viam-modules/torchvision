from dataclasses import dataclass


@dataclass
class Properties:
    implements_classification: bool = False
    implements_detection: bool = False
    implements_get_object_pcd: bool = False
