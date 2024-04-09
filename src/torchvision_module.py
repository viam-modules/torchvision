from typing import ClassVar, List, Mapping, Sequence, Any, Dict, Optional, Union, cast
from viam.media.video import CameraMimeType
from typing_extensions import Self
from viam.components.camera import Camera
from viam.media.video import RawImage
from viam.proto.service.vision import Classification, Detection
from viam.services.vision import Vision, VisionClient
from viam.module.types import Reconfigurable
from viam.resource.types import Model, ModelFamily
from viam.proto.app.robot import ServiceConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.resource.base import ResourceBase
from viam.utils import ValueTypes
from viam.logging import getLogger
from PIL import Image
from src.properties import Properties
import torchvision
from torchvision.models import get_model, get_model_weights, get_weight, list_models
from src.preprocess import Preprocessor
from src.utils import decode_image
import torch
from torch import Tensor
import viam
from torchvision.models import Weights

DETECTION_MODELS: list = list_models(module=torchvision.models.detection)

LOGGER = getLogger(__name__)


class TorchVisionService(Vision, Reconfigurable):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "vision"), "torchvision")

    def __init__(self, name: str):
        super().__init__(name=name)

    @classmethod
    def new_service(
        cls, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        LOGGER.error(f"VIAM VERSION IS {viam.__version__}")
        service = cls(config.name)
        service.reconfigure(config, dependencies)
        return service

    # Validates JSON Configuration
    @classmethod
    def validate_config(cls, config: ServiceConfig) -> Sequence[str]:
        camera_name = config.attributes.fields["camera_name"].string_value
        if camera_name == "":
            raise Exception(
                "A camera name is required for face_identification vision service module."
            )
        return [camera_name]

    def reconfigure(
        self, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        self.camera_name = config.attributes.fields["camera_name"].string_value
        self.camera = cast(
            dependencies[Camera.get_resource_name(self.camera_name)], VisionClient
        )

        def get_attribute_from_config(attribute_name: str, default, of_type=None):
            if attribute_name not in config.attributes.fields:
                return default

            if default is None:
                if of_type is None:
                    raise Exception(
                        "If default value is None, of_type argument can't be empty"
                    )
                type_default = of_type
            else:
                type_default = type(default)

            if type_default == bool:
                return config.attributes.fields[attribute_name].bool_value
            elif type_default == int:
                return int(config.attributes.fields[attribute_name].number_value)
            elif type_default == float:
                return config.attributes.fields[attribute_name].number_value
            elif type_default == str:
                return config.attributes.fields[attribute_name].string_value
            elif type_default == dict:
                return dict(config.attributes.fields[attribute_name].struct_value)

        model_name = get_attribute_from_config("model_name", None, str)
        if model_name in DETECTION_MODELS:
            self.properties.implements_classification = True

        weights = get_attribute_from_config("weights", "DEFAULT")
        try:
            self.model = get_model(model_name, weights=weights)

        except KeyError:
            raise KeyError(
                f"weights: {weights} are not availble for model: {model_name}"
            )
        all_weights = get_model_weights(model_name)
        # self.weights = all_weights.__getattribute__(weights)
        self.weights: Weights = getattr(all_weights, weights)
        self.preprocessor = Preprocessor(weights_transform=self.weights.transforms())
        self.model.eval()
        self.properties = Properties(
            implements_classification=True,
            implements_detection=True,
            implements_get_object_pcd=False,
        )

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[PointCloudObject]:
        if not self.properties.implements_get_object_pcd:
            raise NotImplementedError
        else:
            return 1

    async def get_detections(
        self,
        image: Union[Image.Image, RawImage],
        *,
        extra: Mapping[str, Any],
        timeout: float,
    ) -> List[Detection]:
        if not self.properties.implements_detection:
            raise NotImplementedError

        # img = decode_image(image)
        with torch.no_grad():
            output = self.model(self.preprocessor(image))

        return output

    async def get_classifications(
        self,
        image: Union[Image.Image, RawImage],
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        if not self.properties.implements_classification:
            raise NotImplementedError

        input_tensor = self.preprocessor(image)
        # input_tensor = image
        # torchvision.utils.save_image(input_tensor, "./input.jpg")
        # img = decode_image(image)
        with torch.no_grad():
            prediction: Tensor = self.model(input_tensor)

        prediction = prediction.squeeze(0)
        prediction = prediction.softmax(0)
        scores, top_indices = torch.topk(prediction, k=count)

        category_names = [
            self.weights.meta["categories"][index] for index in top_indices
        ]
        res = [
            Classification(class_name=name, confidence=score)
            for name, score in zip(category_names, scores)
        ]
        return res

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        if not self.properties.implements_classification:
            raise NotImplementedError
        im = await self.camera.get_image(mime_type=CameraMimeType.JPEG)
        image = decode_image(im)
        input_tensor = self.preprocessor(image)
        # input_tensor = image
        # torchvision.utils.save_image(input_tensor, "./input.jpg")
        # img = decode_image(image)
        with torch.no_grad():
            prediction: Tensor = self.model(input_tensor)

        prediction = prediction.squeeze(0)
        prediction = prediction.softmax(0)
        scores, top_indices = torch.topk(prediction, k=count)

        category_names = [
            self.weights.meta["categories"][index] for index in top_indices
        ]
        res = [
            Classification(class_name=name, confidence=score)
            for name, score in zip(category_names, scores)
        ]
        return res

    async def get_detections_from_camera(
        self, camera_name: str, *, extra: Mapping[str, Any], timeout: float
    ) -> List[Detection]:
        if not self.properties.implements_detection:
            raise NotImplementedError

        else:
            raise NotImplementedError

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        raise NotImplementedError
