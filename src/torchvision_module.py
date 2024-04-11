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
from viam.logging import getLogger
from PIL import Image

import torchvision
from torchvision.models import get_model, get_model_weights, list_models
from src.preprocess import Preprocessor
from src.properties import Properties
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
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.camera_name = config.attributes.fields["camera_name"].string_value
        try:
            self.camera = dependencies[Camera.get_resource_name(self.camera_name)]
        except:
            pass

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
            elif type_default == list:
                return list(config.attributes.fields[attribute_name].list_value)
            elif type_default == dict:
                return dict(config.attributes.fields[attribute_name].struct_value)

        model_name = get_attribute_from_config("model_name", None, str)
        self.properties = Properties(
            implements_classification=True,
            implements_detection=False,
            implements_get_object_pcd=False,
        )
        if model_name in DETECTION_MODELS:
            self.properties.implements_classification = False
            self.properties.implements_detection = True

        weights = get_attribute_from_config("weights", "DEFAULT")
        try:
            self.model = get_model(model_name, weights=weights)
        except KeyError:
            raise KeyError(
                f"weights: {weights} are not availble for model: {model_name}"
            )
        all_weights = get_model_weights(model_name)
        self.weights: Weights = getattr(all_weights, weights)
        input_size = get_attribute_from_config("input_size", None, list)
        mean_rgb = get_attribute_from_config("mean_rgb", None, list)
        std_rgb = get_attribute_from_config("std_rgb", None, list)
        use_weight_transform = get_attribute_from_config("use_weight_transform", True)
        swap_r_and_b = get_attribute_from_config("swap_r_and_b", False)
        channel_last = get_attribute_from_config("channel_last", False)

        self.preprocessor = Preprocessor(
            use_weight_transform=use_weight_transform,
            weights_transform=self.weights.transforms(),
            input_size=input_size,
            normalize=(mean_rgb, std_rgb),
            swap_R_and_B=swap_r_and_b,
            channel_last=channel_last,
        )

        self.model.eval()

        self.labels_confidences = get_attribute_from_config(
            "labels_confidences", dict()
        )
        self.default_minimum_confidence = get_attribute_from_config(
            "default_minimum_confidence", 0
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
        input_tensor = self.preprocessor(image)
        with torch.no_grad():
            prediction: Tensor = self.model(input_tensor)[0]
        labels = [self.weights.meta["categories"][i] for i in prediction["labels"]]
        scores = prediction["scores"]
        boxes = prediction["boxes"].to(torch.int64).tolist()
        res = [
            Detection(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                confidence=score,
                class_name=label,
            )
            for (x_min, y_min, x_max, y_max), score, label in zip(boxes, scores, labels)
        ]
        res = self.filter_output(res)
        return res

    async def get_classifications(
        self,
        image: Union[Image.Image, RawImage],
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        if not self.properties.implements_classification:
            return NotImplementedError

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
        res = self.filter_output(res)
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
        res = self.filter_output(res)
        return res

    async def get_detections_from_camera(
        self, camera_name: str, *, extra: Mapping[str, Any], timeout: float
    ) -> List[Detection]:
        if not self.properties.implements_detection:
            raise NotImplementedError
        im = await self.camera.get_image(mime_type=CameraMimeType.JPEG)
        image = decode_image(im)
        input_tensor = self.preprocessor(image)
        with torch.no_grad():
            prediction: Tensor = self.model(input_tensor)[0]
        labels = [self.weights.meta["categories"][i] for i in prediction["labels"]]
        scores = prediction["scores"]
        boxes = prediction["boxes"].to(torch.int64).tolist()
        res = [
            Detection(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                confidence=score,
                class_name=label,
            )
            for (x_min, y_min, x_max, y_max), score, label in zip(boxes, scores, labels)
        ]
        res = self.filter_output(res)
        return res

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        raise NotImplementedError

    def filter_output(
        self, outputs: Union[List[Detection], List[Classification]]
    ) -> Union[List[Detection], List[Classification]]:
        res = []
        for detection in outputs:
            if detection.class_name in self.labels_confidences:
                threshold = self.labels_confidences[detection.class_name]
            else:
                threshold = self.default_minimum_confidence

            if detection.confidence > threshold:
                res.append(detection)

        return res
