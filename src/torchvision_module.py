"""Module that defines the Vision Service that wraps torchvision functionality"""

from typing import ClassVar, List, Mapping, Sequence, Any, Dict, Optional, Union
from typing_extensions import Self
from viam.components.camera import Camera
from viam.media.video import ViamImage, CameraMimeType
from viam.proto.service.vision import Classification, Detection
from viam.services.vision import Vision, CaptureAllResult
from viam.module.types import Reconfigurable
from viam.resource.types import Model, ModelFamily
from viam.proto.app.robot import ServiceConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.resource.base import ResourceBase
from viam.utils import ValueTypes
from viam.logging import getLogger

from PIL import Image
import torch
from torch import Tensor
import torchvision
from torchvision.models import get_model, get_model_weights, list_models
from torchvision.models import Weights
from src.preprocess import Preprocessor
from src.properties import Properties
from src.utils import decode_image

LOGGER = getLogger(__name__)

DETECTION_MODELS: list = list_models(module=torchvision.models.detection)

class TorchVisionService(Vision, Reconfigurable):
    """Torchvision Service class definition"""
    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "vision"), "torchvision")

    def __init__(self, name: str):
        super().__init__(name=name)
        self.camera_name = ""
        self.camera = None

    @classmethod
    def new_service(
        cls, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """create and configure new instance"""
        service = cls(config.name)
        service.reconfigure(config, dependencies)
        return service

    @classmethod
    def validate_config(cls, config: ServiceConfig) -> Sequence[str]:
        """Validates JSON Configuration"""
        model_name = config.attributes.fields["model_name"].string_value
        camera_name = config.attributes.fields["camera_name"].string_value

        if model_name == "":
            raise Exception(
                "A model name is required for this vision service module."
            )
        if camera_name == "":
            raise Exception(
                "A camera name is required for this vision service module."
            )
        return [camera_name]

    def reconfigure(
        self, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        """Handles attribute reconfiguration"""
        self.dependencies = dependencies
        self.camera_name = config.attributes.fields["camera_name"].string_value
        self.camera = self.dependencies[Camera.get_resource_name(self.camera_name)]

        # pylint: disable=too-many-return-statements, inconsistent-return-statements
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
            if type_default == int:
                return int(config.attributes.fields[attribute_name].number_value)
            if type_default == float:
                return config.attributes.fields[attribute_name].number_value
            if type_default == str:
                return config.attributes.fields[attribute_name].string_value
            if type_default == list:
                return list(config.attributes.fields[attribute_name].list_value)
            if type_default == dict:
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
        except KeyError as e:
            raise KeyError(
                f"weights: {weights} are not availble for model: {model_name}"
            ) from e
        all_weights = get_model_weights(model_name)
        self.weights: Weights = getattr(all_weights, weights)
        input_size = get_attribute_from_config("input_size", None, list)
        if input_size is not None:
            input_size = [int(size) for size in input_size]
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
            swap_r_and_b=swap_r_and_b,
            channel_last=channel_last,
        )

        self.model.eval()

        self.labels_confidences = get_attribute_from_config(
            "labels_confidences", {}
        )
        self.default_minimum_confidence = get_attribute_from_config(
            "default_minimum_confidence", 0
        )
    #pylint: disable=too-many-arguments,too-many-positional-arguments
    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        """Get the next image, detections, classifications, and objects all together,
        given a camera name. Used for visualization.

        ::

            camera_name = "cam1"

            # Grab the detector you configured on your machine
            my_detector = VisionClient.from_robot(robot, "my_detector")

            # capture all from the next image from the camera
            result = await my_detector.capture_all_from_camera(
                camera_name,
                return_image=True,
                return_detections=True,
            )

        Args:
            camera_name (str): The name of the camera to use for detection
            return_image (bool): 
                Ask the vision service to return the camera's latest image
            return_classifications (bool): 
                Ask the vision service to return its latest classifications
            return_detections (bool): 
                Ask the vision service to return its latest detections
            return_object_point_clouds (bool): 
                Ask the vision service to return its latest 3D segmentations

        Returns:
            vision.CaptureAllResult: 
                A class that stores all potential returns from the vision service.
            It can  return the image from the camera along with its associated detections, 
            classifications, and objects, as well as any extra info the model may provide.
        """
        result = CaptureAllResult()

        if camera_name not in (self.camera_name, ""):
            raise ValueError(
                "Camera name passed to method:",
                camera_name,
                "is not the configured 'camera_name'",
                self.camera_name,
            )
        image = await self.get_image_from_dependency(camera_name)

        if return_image:
            result.image = image
        if return_classifications:
            try:
                classifications = await self.get_classifications(image, 1)
                result.classifications = classifications
            # pylint: disable=broad-exception-caught
            except Exception as e:
                LOGGER.info(f"getClassifications failed: {e}")
        if return_detections:
            try:
                detections = await self.get_detections(image, timeout=timeout, extra=None)
                result.detections = detections
            # pylint: disable=broad-exception-caught
            except Exception as e:
                LOGGER.info(f"getDetections failed: {e}")

        return result

    # pylint: disable=missing-function-docstring
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
        return 1

    async def get_detections(
        self,
        image: Union[Image.Image, ViamImage],
        *,
        extra: Mapping[str, Any],
        timeout: float,
    ) -> List[Detection]:
        """Get detections from an image"""
        if not self.properties.implements_detection:
            raise NotImplementedError
        LOGGER.info(f"input image is: {type(image)}")
        image = decode_image(image)
        input_tensor = self.preprocessor(image)
        with torch.no_grad():
            prediction: Tensor = self.model(input_tensor)[0]
        return self.wrap_detections(prediction)

    async def get_classifications(
        self,
        image: Union[Image.Image, ViamImage],
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        """Get classifications from image"""
        if not self.properties.implements_classification:
            return NotImplementedError
        image = decode_image(image)
        input_tensor = self.preprocessor(image)
        with torch.no_grad():
            prediction: Tensor = self.model(input_tensor)
        out = self.wrap_classifications(prediction, count)
        LOGGER.info(f"output: {type(out)}, {out}")
        return out

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        """Gets classifications from a camera dependency"""
        if not self.properties.implements_classification:
            raise NotImplementedError

        if camera_name not in (self.camera_name, ""):
            raise ValueError(
                "Camera name passed to method:",
                camera_name,
                "is not the configured 'camera_name'",
                self.camera_name,
            )
        image = await self.get_image_from_dependency(camera_name)
        input_tensor = self.preprocessor(image)
        with torch.no_grad():
            prediction: Tensor = self.model(input_tensor)
        return self.wrap_classifications(prediction, count)

    async def get_detections_from_camera(
        self, camera_name: str, *, extra: Mapping[str, Any] = None, timeout: float = None,
    ) -> List[Detection]:
        """Gets detections from a camera dependency"""
        if not self.properties.implements_detection:
            raise NotImplementedError
        if camera_name not in (self.camera_name, ""):
            raise ValueError(
                "Camera name passed to method:",
                camera_name,
                "is not the configured 'camera_name'",
                self.camera_name,
            )
        image = await self.get_image_from_dependency(camera_name)
        LOGGER.info(f"input image is: {type(image)}")
        input_tensor = self.preprocessor(image)
        with torch.no_grad():
            prediction: Tensor = self.model(input_tensor)[0]
        return self.wrap_detections(prediction)

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Properties:
        """
        Get info about what vision methods the vision service provides. 
        Currently returns boolean values that
        state whether the service implements the classification, detection, 
        and/or 3D object segmentation methods.

        ::
                # Grab the detector you configured on your machine
                my_detector = VisionClient.from_robot(robot, "my_detector")
                properties = await my_detector.get_properties()
                properties.detections_supported      # returns True
                properties.classifications_supported # returns False

        Returns:
            Properties: The properties of the vision service
        """
        return self.properties

    # pylint: disable=missing-function-docstring
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

    async def get_image_from_dependency(self, camera_name: str):
        # cam = self.dependencies[Camera.get_resource_name("")]
        im = await self.camera.get_image(mime_type=CameraMimeType.JPEG)
        return decode_image(im)

    def wrap_detections(self, prediction: dict):
        """_summary_
        converts prediction output tensor from torchvision model
        for viam API
        Args:
            prediction (dict): _description_

        Returns:
            _type_: _description_
        """
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

    def wrap_classifications(self, prediction, count):
        """_summary_
        Converts prediction output tensor from torchvision model
        for viam API
        Args:
            prediction (Tensor):
            count (int):
        """
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
