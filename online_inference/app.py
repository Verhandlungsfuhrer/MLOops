from logging import getLogger

import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.exceptions import FastAPIError
from fastapi.responses import Response
from PIL import Image
from tritonclient.http import InferenceServerClient, InferInput

app = FastAPI()
logger = getLogger()


@app.post("/infer")
async def infer_client(image: UploadFile) -> Response:
    if image.filename.endswith((".png", ".jpeg", ".jpg", ".txt")):
        pil_image = Image.open(image.file).convert(mode="RGB")
        image_array = np.array(pil_image)
        image_array = image_array[None, ...]
        inp = InferInput("RAW_IMAGE", image_array.shape, "UINT8")
        inp.set_data_from_numpy(image_array)
        out = app.triton_client.infer("ensemble", inputs=[inp])
        result = out.as_numpy("LABEL")
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
