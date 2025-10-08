import numpy as np

import nvidia.dali as dali
from nvidia.dali.plugin.triton import autoserialize
from nvidia.dali.types import FLOAT


MEAN = np.array([0.485, 0.456, 0.406])[:, None, None]
STD = np.array([0.229, 0.224, 0.225])[:, None, None]


@autoserialize
@dali.pipeline_def(batch_size=32, num_threads=4, device_id=0)
def pipe():  # type: ignore
    images = dali.fn.external_source(device="cpu", name="DALI_INPUT_0")
    images = dali.fn.resize(images, resize_x=224, resize_y=224)
    images = dali.fn.cast(images / 255, dtype=FLOAT)
    images = dali.fn.transpose(images, perm=[2, 0, 1])
    images = dali.fn.normalize(images, mean=MEAN, stddev=STD)
    return images


saved_pipe = pipe()
saved_pipe.serialize(filename="model.dali")
