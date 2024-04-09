# VIAM FACE IDENTIFICATION MODULE

This is a [Viam module](https://docs.viam.com/extend/modular-resources/) providing a model of vision service for [TorchVision's New Multi-Weight Support API](https://pytorch.org/blog/introducing-torchvision-new-multi-weight-support-api/).

<p align="center">
 <img src="https://pytorch.org/assets/images/torchvision_gif.gif" width=80%, height=70%>
 </p>

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


## Installation with `pip install` 

```
pip install -r requirements.txt
```

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
        "camera_name": "cam"
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


| Name          | Type   | Inclusion    | Default   | Description                                                                                                                                                                        |
| ------------- | ------ | ------------ | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `camera_name` | string | **Required** |           | Camera name to be used a source.                                                                                                                                                   |
| `model_name`  | string | **Required** |           | Vision model name as expected by the method [get_model()](https://pytorch.org/vision/main/models.html#listing-and-retrieving-available-models) from torchvision multi-weight API.  |
| `weights`     | string | Optional     | `DEFAULT` | Weights model name as expected by the method [get_model()](https://pytorch.org/vision/main/models.html#listing-and-retrieving-available-models) from torchvision multi-weight API. |


## RESSOURCES
- [Table of all available classification weights](https://pytorch.org/vision/main/models.html#table-of-all-available-classification-weights)
- [Quantized models](https://pytorch.org/vision/main/models.html#quantized-models)
