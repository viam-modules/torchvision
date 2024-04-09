from typing import ClassVar, List, Mapping, Sequence, Any, Dict, Optional, Union
from viam.media.video import CameraMimeType
from typing_extensions import Self
from viam.components.camera import Camera
from viam.media.video import RawImage
from viam.proto.service.vision import Classification, Detection
from viam.services.vision import Vision
from viam.module.types import Reconfigurable
from viam.resource.types import Model, ModelFamily
from viam.proto.app.robot import ServiceConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.resource.base import ResourceBase
from viam.utils import ValueTypes

from PIL import Image
from properties import Properties
import torchvision
from torchvision.models import get_model, get_model_weights, get_weight, list_models
from torchvision_module import TorchVisionService
from viam.utils import dict_to_struct
import asyncio
import numpy as np


async def main():
    tvs = TorchVisionService(name="torch_vision_service")
    # tvs.model = get_model("resnet18")
    cfg = ServiceConfig(attributes=dict_to_struct({"model_name": "resnet50"}))
    # cfg.attributes.fields["model_name"] = "resnet18"

    tvs.reconfigure(config=cfg, dependencies=None)
    path_to_input_image = "/Users/robinin/torch-infer/torchvision/src/grass_hopper.jpg"
    input_image = np.array(Image.open(path_to_input_image))
    # input_image = torchvision.io.read_image(path_to_input_image)
    output = await tvs.get_classifications(image=input_image, extra=None, count=3)

    print(output)


if __name__ == "__main__":
    asyncio.run(main())
