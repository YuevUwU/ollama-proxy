import logging

import lmstudio as lms
from fastapi import FastAPI, HTTPException, Request
from starlette.responses import Response

import api_models
import helpers
import lms_rest_api

FAKE_OLLAMA_VER = "0.15.5"
LMSTUDIO_API_HOST = "http://localhost:1234/"

# LMStudio REST API
lms_api_client = lms_rest_api.LMStudioRestAPIClient(LMSTUDIO_API_HOST)

# LMStudio Python SDK
lms.configure_default_client(LMSTUDIO_API_HOST)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Run it with uvicorn
app = FastAPI()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log incoming requests.
    """
    body = await request.body()
    logging.info(
        f"📡 Incoming Request: {request.method} {request.url} | Payload: {body.decode('utf-8')}"
    )

    response: Response = await call_next(request)

    logging.info(
        f"📡 Response: {response.status_code} for {request.method} {request.url}"
    )
    return response


@app.post("/api/generate")
async def generate_completion():
    raise NotImplementedError


@app.post("/api/create")
async def create_model():
    raise NotImplementedError


@app.head("/api/blobs/:digest")
async def check_if_blob_exist():
    raise NotImplementedError


@app.post("/api/blobs/:digest")
async def push_blob():
    raise NotImplementedError


@app.get("/api/tags")
async def list_local_models():
    return await helpers.list_lms_models(lms_api_client, loaded_only=False)
    # return await _list_lms_models(loaded_only=True)
    # return {"models":[{"name":"hf.co/lmstudio-community/Qwen3-8B-GGUF:Q6_K","model":"hf.co/lmstudio-community/Qwen3-8B-GGUF:Q6_K","modified_at":"2026-02-06T21:48:37.463632+08:00","size":6725903011,"digest":"024cd4933b036562bf9998377df3e0d825f124dcffc96cecf86a7e29de32bce2","details":{"parent_model":"","format":"gguf","family":"qwen3","families":["qwen3"],"parameter_size":"8.2B","quantization_level":"Q6_K"}},{"name":"hf.co/mradermacher/Ministral-3-8B-Base-2512-GGUF:Q5_K_M","model":"hf.co/mradermacher/Ministral-3-8B-Base-2512-GGUF:Q5_K_M","modified_at":"2026-02-06T21:35:57.2494282+08:00","size":6058739531,"digest":"7db1049116b1c094deefcc12424f5e0b57d477f345d8621557b39e246f7df527","details":{"parent_model":"","format":"gguf","family":"mistral3","families":["mistral3"],"parameter_size":"8.5B","quantization_level":"Q5_K_M"}}]}


@app.post("/api/show")
async def show_model_info(req: api_models.ShowRequest) -> api_models.ShowResponse:
    if req.verbose is True:
        raise NotImplementedError

    list_resp = await helpers.list_lms_models(
        lms_api_client, loaded_only=False, specific_model=req.model
    )
    if len(list_resp.models) == 0:
        raise HTTPException(404, f"model '{req.model}' not found")
    if len(list_resp.models) > 1:
        raise RuntimeError(f"Model name {req.model} has multiple models")
    model = list_resp.models[0]
    return api_models.ShowResponse(details=model.details)


@app.post("/api/copy")
async def copy_model():
    raise NotImplementedError


@app.delete("/api/delete")
async def delete_model():
    raise NotImplementedError


@app.post("/api/pull")
async def pull_model():
    raise NotImplementedError


@app.post("/api/push")
async def push_model():
    raise NotImplementedError


@app.post("/api/embed")
async def generate_embedding():
    raise NotImplementedError


@app.get("/api/ps")
async def list_running_models():
    return await helpers.list_lms_models(lms_api_client, loaded_only=True)


@app.post("/api/embeddings")
async def generate_embedding_supereded():
    raise NotImplementedError


@app.get("/api/version")
async def version() -> api_models.VersionResponse:
    return api_models.VersionResponse(version=FAKE_OLLAMA_VER)


@app.get("/")
async def root():
    logging.info("🟢 Root endpoint accessed")
    return {"message": "Ollama Proxy to LM Studio is running"}
