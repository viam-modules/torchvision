from viam.proto.app.robot import ServiceConfig

from PIL import Image
from src.torchvision_module import TorchVisionService
from viam.utils import dict_to_struct
import asyncio
import numpy as np


async def main():
    path_to_input_image = "/Users/robinin/torch-infer/torchvision/src/grass_hopper.jpg"
    input_image = np.array(Image.open(path_to_input_image))
    tvs = TorchVisionService(name="torch_vision_service")
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
    output = tvs.reconfigure(config=cfg, dependencies=None)

    output = await tvs.get_classifications(
        image=input_image, extra=None, timeout=0, count=10
    )
    print(f"output of get_classifications: {output}\n")

    ##Config for a detector
    cfg = ServiceConfig(
        attributes=dict_to_struct(
            {
                "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
                "labels_confidences": {},
            }
        )
    )

    output = tvs.reconfigure(config=cfg, dependencies=None)
    output = await tvs.get_detections(
        image=input_image,
        extra=None,
        timeout=0,
    )
    print(f"output of get_detections: {output}")


if __name__ == "__main__":
    asyncio.run(main())
