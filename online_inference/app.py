from logging import getLogger

from fastapi import FastAPI, UploadFile
from fastapi.responses import Response
from fastapi.exceptions import FastAPIError
from tritonclient.http import InferenceServerClient, InferInput
from PIL import Image
import numpy as np

app = FastAPI()
logger = getLogger()


@app.post("/infer")
async def infer_client(image: UploadFile) -> Response:
    if image.filename.endswith((".png", ".jpeg", ".jpg")):
        pil_image = Image.open(image.file)
        image_array = np.array(pil_image)
        image_array = image_array.astype("float32") / 255
        image_array = (image_array - 0.5) / 0.5
        if len(image_array.shape) != 3:
            return Response("Expect color image", status_code=400)
        elif image_array.shape[-1] != 3:
            return Response("Expect color image without alpha chanel", status_code=400)
        elif image_array.shape[0] != 224:
            return Response("Expect height of image equal 224", status_code=400)
        elif image_array.shape[1] != 224:
            return Response("Expect width of image equal 224", status_code=400)
        image_array = image_array.transpose(2, 0, 1)[None, ...]
        inp = InferInput("image", [1, 3, 224, 224], "FP32")
        inp.set_data_from_numpy(image_array)
        out = app.triton_client.infer("image_clf", inputs=[inp])
        result = out.as_numpy("label")
        logger.info(result.argmax())
        return Response(app.labels[result.argmax()], status_code=200)
    else:
        return Response("Wrong image extension", status_code=400)


@app.get("/")
async def health() -> Response:
    return Response("server is live", status_code=200)


@app.on_event("startup")
async def connect_to_triton() -> None:
    app.triton_client = InferenceServerClient("triton:8000")
    with open("labels.txt") as f:
        app.labels = [line.strip() for line in f.readlines()]
    status = app.triton_client.is_server_live()
    if not status:
        raise FastAPIError("inference client is not working")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8190)
