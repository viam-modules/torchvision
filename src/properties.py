"""Simple Class definition for Properties"""
from dataclasses import dataclass

# pylint: disable=missing-class-docstring
@dataclass
class Properties:
    implements_classification: bool = False
    implements_detection: bool = False
    implements_get_object_pcd: bool = False
