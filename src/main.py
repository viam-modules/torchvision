"""Main entrypoint file to run module"""

import asyncio

from viam.services.vision import Vision
from viam.module.module import Module
from viam.resource.registry import Registry, ResourceCreatorRegistration
from src.torchvision_module import TorchVisionService


async def main():
    """
    This function creates and starts a new module, after adding all desired
    resource models. Resource creators must be registered to the resource
    registry before the module adds the resource model.
    """
    Registry.register_resource_creator(
        Vision.SUBTYPE,
        TorchVisionService.MODEL,
        ResourceCreatorRegistration(
            TorchVisionService.new_service,
            TorchVisionService.validate_config,
        ),
    )
    module = Module.from_args()

    module.add_model_from_registry(Vision.SUBTYPE, TorchVisionService.MODEL)
    await module.start()


if __name__ == "__main__":
    asyncio.run(main())
