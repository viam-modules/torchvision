from typing import Any, List, Tuple
from torchvision.transforms import v2
import numpy as np
from numpy.typing import NDArray
import torchvision.transforms.v2.functional as F
import torch


class SwapRandB:
    def __call__(self, rgb):
        if rgb.ndim == 3:
            bgr = F.permute_channels(rgb, permutation=[2, 1, 0])
        else:
            raise RuntimeError
        return bgr

    def __repr__(self):
        return self.__class__.__name__ + "()"


class ToChannelsLast:
    def __call__(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        elif x.ndim != 4:
            raise RuntimeError("non")
        return x.to(memory_format=torch.channels_last)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Preprocessor:
    def __init__(
        self,
        weights_transform=None,
        use_weight_transform=False,
        channel_last: bool = False,
        swap_R_and_B: bool = False,  # should be use if you have an RGB image and model was trained with OpenCV Dataloader or if you open yoiur image using OpenCV
        normalize: Tuple = None,  # eg. ([mean_r, mean_g, mean_b], [std_r, std_g, std_b]) ALWAYS RGB ORDER # type: ignore
        input_size: List[int] = None,
    ) -> None:
        self.input_size = input_size
        pipeline = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        if weights_transform is not None and use_weight_transform:
            pipeline.append(weights_transform)

        if input_size is not None:
            pipeline.append(v2.Resize(size=input_size))

        if normalize[0] is not None and normalize[1] is not None:
            pipeline.append(v2.Normalize(mean=normalize[0], std=normalize[1]))
        if swap_R_and_B:
            pipeline.append(SwapRandB())
        if channel_last:
            pipeline.append(ToChannelsLast())

        self.transform = v2.Compose(pipeline)

    def __call__(
        self,
        img: np.ndarray,
    ) -> Any:
        img = self.transform(img)

        if img.ndim == 3:
            return img.unsqueeze(0)
        return img  # add batch dim
