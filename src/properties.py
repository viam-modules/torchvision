"""Simple Class definition for Properties"""
from dataclasses import dataclass

# pylint: disable=missing-class-docstring
@dataclass
class Properties:
    classifications_supported: bool = False
    detections_supported: bool = False
    object_point_clouds_supported: bool = False
