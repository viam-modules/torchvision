# Viam Torchvision Module


This is a [Viam module](https://docs.viam.com/extend/modular-resources/) providing a model of vision service for [TorchVision's New Multi-Weight Support API](https://pytorch.org/blog/introducing-torchvision-new-multi-weight-support-api/).
<p align="center">
 <img src="https://pytorch.org/assets/images/torchvision_gif.gif" width=80%, height=70%>
 </p>


 For a given model architecture (e.g. *ResNet50*), multiple weights can be available and each of those weights comes with Metadata (preprocessing and labels). 
## Getting started

To use this module, follow these instructions to [add a module from the Viam Registry](https://docs.viam.com/modular-resources/configure/#add-a-module-from-the-viam-registry) and select the `viam:vision:torchvision` model from the [`torchvision` module](https://app.viam.com/module/viam/torchvision).
Depending on the type of models configured, the module implements:

- For detectors:
  - `GetDetections()`
  - `GetDetectionsFromCamera()`

- For classifiers:
  - `GetClassifications()`
  - `GetClassificationsFromCamera()`

> [!NOTE]  
>See [vision service API](https://docs.viam.com/services/vision/#api) for more details.

## Configure your `torchvision` vision service

> [!NOTE]  
> Before configuring your vision service, you must [create a robot](https://docs.viam.com/manage/fleet/robots/#add-a-new-robot).

Navigate to the **Config** tab of your robot’s page in [the Viam app](https://app.viam.com/). Click on the **Services** subtab and click **Create service**. Select the `Vision` type, then select the `torchvision` model. Enter a name for your service and click **Create**.

### Example of config with a camera and transform camera

```json
{
  "modules": [
    {
      "executable_path": "/path/to/run.sh",
      "name": "mytorchvisionmodule",
      "type": "local"
    }
  ],
  "services": [
    {
      "attributes": {
        "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "labels_confidences": {"grasshopper": 0.5, 
                                "cricket": 0.45 },
        "default_minimum_confidence": 0.3
        
      },
      "name": "detector-module",
      "type": "vision",
      "namespace": "rdk",
      "model": "viam:vision:torchvision"
    }
  ],
    "components": [
    {
      "namespace": "rdk",
      "attributes": {
        "video_path": "video0"
      },
      "depends_on": [],
      "name": "cam",
      "model": "webcam",
      "type": "camera"
    },
    {
      "model": "transform",
      "type": "camera",
      "namespace": "rdk",
      "attributes": {
        "source": "cam",
        "pipeline": [
          {
            "attributes": {
              "detector_name": "detector-module",
              "confidence_threshold": 0.5
            },
            "type": "detections"
          }
        ]
      },
      "depends_on": [],
      "name": "detections"
    }
  ]
}
```


### Attributes description

The following attributes are available to configure your deepface module:


| Name | Type | Inclusion | Default | Description |
| ---- | ---- | --------- | ------- | ----------- ||
| `model_name`                 | string                | **Required** |             | Vision model name as expected by the method [get_model()](https://pytorch.org/vision/main/models.html#listing-and-retrieving-available-models) from torchvision multi-weight API.                                                                                                                                                                  |
| `weights`                    | string                | Optional     | `DEFAULT`   | Weights model name as expected by the method [get_model()](https://pytorch.org/vision/main/models.html#listing-and-retrieving-available-models) from torchvision multi-weight API.                                                                                                                                                                 |
| `default_minimum_confidence` | float                 | Optional     |             | Default minimum confidence for filtering all labels that are not specified in `label_confidences`.                                                                                                                                                                                                                                                 |
| `labels_confidences`         | dict[str, float]      | Optional     |             | Dictionary specifying minimum confidence thresholds for specific labels. Example: `{"grasshopper": 0.5, "cricket": 0.45}`. If a label has a confidence set lower that `default_minimum_confidence`, that confidence over-writes the default for the specified label if `labels_confidences` is left blank, no filtering on labels will be applied. |
| `use_weight_transform`       | bool                  | Optional     | True        | Loads preprocessing transform from weights metadata.                                                                                                                                                                                                                                                                                               |
| `input size`                 | List[int]             | Optional     | `None`      | Resize the image. Overides resize from weights metadata.                                                                                                                                                                                                                                                                                           |
| `mean_rgb`                   | [float, float, float] | Optional     | `[0, 0, 0]` | Specifies the mean and standard deviation values for normalization in RGB order                                                                                                                                                                                                                                                                    |
| `std_rgb`                    | [float, float, float] | Optional     | `[1, 1, 1]` | Specifies the standard deviation values for normalization in RGB order.                                                                                                                                                                                                                                                                            |
| `swap_r_and_b`               | bool                  | Optional     | `False`     | If True, swaps the R and B channels in the input image. Use this if the images passed as inputs to the model are in the OpenCV format.                                                                                                                                                                                                             |
| `channel_last`               | bool                  | Optional     | `False`     | If True, the image tensor will be converted to channel-last format. Default is False.                                                                                                                                                                                                                                                              |

### Preprocessing transforms behavior and **order**:
   - If there are a transform in the metadata of the weights and `use_weight_transform` is True, `weights_transform` is added to the pipeline.
   - If `input_size` is provided, the image is resized using `v2.Resize()` to the specified size.
   - If both mean and standard deviation values are provided in `normalize`, the image is normalized using `v2.Normalize()` with the specified mean and standard deviation values.
   - If `swap_R_and_B` is set to `True`, first and last channel are swapped. 
   - If `channel_last` is `True`, a transformation is applied to conv.ert the channel order to the last dimension format. (C, H ,W) -> (H, W, X).



## RESSOURCES
- [Table of all available classification weights](https://pytorch.org/vision/main/models.html#table-of-all-available-classification-weights)
- [Quantized models](https://pytorch.org/vision/main/models.html#quantized-models)
