from viam.proto.app.robot import ServiceConfig

from PIL import Image
from viam.utils import dict_to_struct
import asyncio
import numpy as np

import pytest
from typing import Any, Mapping

from grpclib.testing import ChannelFor
from PIL import Image

from viam.media.utils.pil import pil_to_viam_image
from viam.media.video import CameraMimeType, ViamImage
from viam.proto.app.robot import ComponentConfig

from viam.services.vision import Classification, Detection, Vision, VisionClient
from viam.utils import dict_to_struct, struct_to_dict
from viam.resource.manager import ResourceManager
from src.properties import Properties
from src.torchvision_module import TorchVisionService
from src.torchvision_mock import TorchVisionMock
from google.protobuf.struct_pb2 import Struct

path_to_input_image = "tests/grasshopper.jpg"
input_image = np.array(Image.open(path_to_input_image))
cfg = ServiceConfig(
    attributes=dict_to_struct(
        {
            "model_name": "resnet50",
            "weights": "IMAGENET1K_V1",
            "camera_name": "cam",
        }
    ),
        depends_on=["zebi"],
    )

VISION_SERVICE_NAME = "vision1"
DETECTIONS = []
PROPERTIES = Properties(
    implements_classification=True, 
    implements_detection=False, 
    implements_get_object_pcd=False
)

def make_component_config(dictionary: Mapping[str, Any]) -> ComponentConfig:
    struct = Struct()
    struct.update(dictionary=dictionary)
    return ComponentConfig(attributes=struct)

config = (
    make_component_config({
        "model_name": "resnet_50"
    }),
    "received only one dimension attribute"
)

@pytest.fixture(scope="function")
def vision() -> TorchVisionService:
    return TorchVisionService(
        name='tvs'
    )

@pytest.fixture(scope="function")
def vision_mock() -> TorchVisionMock:
    return TorchVisionMock(
        name='tvs',
        image=input_image
    )

class TestVision:
    @pytest.mark.asyncio
    async def test_validate(self):
        response = TorchVisionService.validate_config(config=config[0])

    @pytest.mark.asyncio
    async def test_get_properties(self, vision: TorchVisionService):
        vision.reconfigure(cfg, dependencies=None)
        response = await vision.get_properties()
        assert response == PROPERTIES

    @pytest.mark.asyncio
    async def test_capture_all_from_camera(self, vision_mock: TorchVisionMock):
        vision_mock.reconfigure(cfg, dependencies=None)
        response = await vision_mock.capture_all_from_camera(
            "fake-camera",
            return_image=True,
            return_detections=False,
            return_classifications=True,
        )
        assert response.image.data == input_image.data
        # can also test mimetype here
        assert response.detections is None
        assert response.classifications is not None
        assert response.objects is None
