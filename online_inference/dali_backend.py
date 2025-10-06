import nvidia.dali as dali
from nvidia.dali.plugin.triton import autoserialize


@autoserialize
@dali.pipeline_def(batch_size=256, num_threads=4, device_id=0)
def pipe():  # type: ignore
    images = dali.fn.external_source(device="cpu", name="DALI_INPUT_0")
    images = dali.fn.image_decoder(images, device="mixed")
    images = dali.fn.resize(images, resize_x=224, resize_y=224)
    return images
