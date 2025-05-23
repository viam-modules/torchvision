from viam.proto.app.robot import ServiceConfig

from PIL import Image
from viam.utils import dict_to_struct
import asyncio
import numpy as np

import pytest
from unittest.mock import AsyncMock, Mock, patch
from typing import Any, Mapping

from PIL import Image

from viam.components.camera import Camera
from viam.media.video import CameraMimeType
from viam.media.utils.pil import pil_to_viam_image
from viam.proto.app.robot import ComponentConfig

from viam.utils import dict_to_struct
from src.properties import Properties
from src.torchvision_module import TorchVisionService
from google.protobuf.struct_pb2 import Struct

path_to_input_image = "tests/grasshopper.jpg"
image = Image.open(path_to_input_image)
input_image = np.array(image)
cfg = ServiceConfig(
    attributes=dict_to_struct(
        {
            "model_name": "resnet50",
            "weights": "IMAGENET1K_V1",
            "camera_name": "fake_cam",
        }
    ))

VISION_SERVICE_NAME = "vision1"
DETECTIONS = []
PROPERTIES = Properties(
    classifications_supported=True, 
    detections_supported=False,
    object_point_clouds_supported=False
)

def make_component_config(dictionary: Mapping[str, Any]) -> ComponentConfig:
    struct = Struct()
    struct.update(dictionary=dictionary)
    return ComponentConfig(attributes=struct)

config = make_component_config({
        "model_name": "resnet_50"})


config2 = make_component_config({
        "model_name": "resnet_50",
        "camera_name": "fake_cam"
    })

@pytest.fixture(scope="function")
def vision() -> TorchVisionService:
    return TorchVisionService(
        name='tvs'
    )

class TestVision:
    @pytest.mark.asyncio
    async def test_validate(self):
        with pytest.raises(Exception):
            response = TorchVisionService.validate_config(config=config)
        response = TorchVisionService.validate_config(config=config2)

    @pytest.mark.asyncio
    @patch('viam.components.camera.Camera.get_resource_name', return_value="fake_cam")
    @patch.object(TorchVisionService, 'get_image_from_dependency', new_callable=AsyncMock)
    async def test_get_properties(self, get_image_from_dependency, fake_cam):
        vision =  TorchVisionService(
            name='tvs'
        )
        vision.camera_name = "fake_cam"
        get_image_from_dependency.return_value = input_image

        vision.reconfigure(cfg, dependencies={"fake_cam": Mock()})
        response = await vision.get_properties()
        assert response == PROPERTIES

    @patch('viam.components.camera.Camera.get_resource_name', return_value="fake_cam")
    def test_capture_all_from_camera(self, fake_cam):
        camera =  TorchVisionService(
            name='tvs'
        )
        camera.camera_name = "fake_cam"

        camera.reconfigure(cfg, dependencies={"fake_cam": Mock()})
        camera.camera.get_image = AsyncMock(return_value=pil_to_viam_image(image, mime_type=CameraMimeType.JPEG))

        # without point clouds = True
        result = asyncio.run(camera.capture_all_from_camera(
            'fake_cam',
            return_image=True,
            return_classifications=True,
            return_detections=True
        ))

        # assert result.image.all() == input_image
        assert result.classifications is not None
        assert result.detections is None

        result = asyncio.run(camera.capture_all_from_camera(
            'fake_cam',
            return_image=True,
            return_classifications=True,
            return_detections=True,
            return_object_point_clouds=True
        ))

        # assert result.image == input_image
        assert result.classifications is not None
        assert result.detections is None
        assert result.objects is None
        # mock_get_classifications.assert_called_once_with('test_image', 1)
        # mock_get_detections.assert_called_once_with('test_image', timeout=None)

    @patch('viam.components.camera.Camera.get_resource_name', return_value="fake_cam")
    def test_default_camera_behavior(self, fake_cam):
        vs = TorchVisionService(
            name='tvs'
        )

        vs.reconfigure(cfg, dependencies={"fake_cam": Mock()})
        vs.camera.get_image = AsyncMock(return_value=pil_to_viam_image(image, mime_type=CameraMimeType.JPEG))

        result = vs.get_classifications_from_camera(
            "",
            count=1,
        )
        assert result is not None

        result = asyncio.run(vs.capture_all_from_camera(
            "",
            return_classifications=True,
        ))
        assert result is not None
        assert result.classifications is not None

        with pytest.raises(ValueError) as excinfo:
            asyncio.run(vs.get_classifications_from_camera(
                "not_cam",
                count=1,
            ))
        assert 'not_cam' in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            asyncio.run(asyncio.run(vs.capture_all_from_camera(
                "not_cam",
                return_classifications=True,
                return_detections=True,
            )))
        assert 'not_cam' in str(excinfo.value)

    @patch('viam.components.camera.Camera.get_resource_name', return_value="fake_cam")
    @patch.object(TorchVisionService, 'get_image_from_dependency', new_callable=AsyncMock)
    def test_get_detections(self, get_image_from_dependency, fake_cam):
        vision =  TorchVisionService(
            name='tvs'
        )
        vision.camera_name = "fake_cam"
        get_image_from_dependency.return_value = input_image

        cfg = ServiceConfig(
            attributes=dict_to_struct(
                {
                    "model_name": "fasterrcnn_resnet50_fpn",
                    "camera_name": "fake_cam",
                }
            )
        )

        vision.reconfigure(cfg, dependencies={"fake_cam": Mock()})
        response = asyncio.run(vision.get_detections_from_camera("fake_cam"))
        assert response is not None
        assert response[0].x_min is not None
        assert response[0].y_min is not None
        assert response[0].x_max is not None
        assert response[0].y_max is not None
        assert response[0].x_min_normalized == response[0].x_min / input_image.shape[1]
        assert response[0].y_min_normalized == response[0].y_min / input_image.shape[0]
        assert response[0].x_max_normalized == response[0].x_max / input_image.shape[1]
        assert response[0].y_max_normalized == response[0].y_max / input_image.shape[0]
