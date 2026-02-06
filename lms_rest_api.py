from __future__ import annotations

from typing import List, Literal, Optional

import httpx
from httpx import Response
from pydantic import BaseModel, Field

# ── Response models ────────────────────────────────────────────────────────────
# https://lmstudio.ai/docs/developer/rest/list

class QuantizationInfo(BaseModel):
    name: Optional[str] = None
    bits_per_weight: Optional[float] = None


class LoadedInstanceConfig(BaseModel):
    context_length: int
    eval_batch_size: Optional[int] = None  # absent for embedding models
    flash_attention: Optional[bool] = None
    num_experts: Optional[int] = None
    offload_kv_cache_to_gpu: Optional[bool] = None


class LoadedInstance(BaseModel):
    id: str
    config: LoadedInstanceConfig


class Capabilities(BaseModel):
    vision: bool
    trained_for_tool_use: bool


class ModelInfo(BaseModel):
    type: Literal["llm", "embedding"]
    publisher: str
    key: str
    display_name: str
    architecture: Optional[str] = None  # null for embedding
    quantization: Optional[QuantizationInfo] = None
    size_bytes: int
    params_string: Optional[str] = None
    loaded_instances: List[LoadedInstance] = Field(default_factory=list)
    max_context_length: int
    format: Optional[Literal["gguf", "mlx"]] = None
    capabilities: Optional[Capabilities] = None  # absent for embedding
    description: Optional[str] = None


class ModelsResponse(BaseModel):
    models: List[ModelInfo]


# ── Client ─────────────────────────────────────────────────────────────────────


class LMStudioRestAPIClient:
    def __init__(
        self,
        base_url,
        timeout: float = 15.0,
    ):
        self.base_url = base_url.rstrip("/")

        headers = {"Accept": "application/json"}

        self.http = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
        )

    async def close(self) -> None:
        await self.http.aclose()

    async def list_models(self) -> ModelsResponse:
        """
        GET /api/v1/models
        Returns list of available models (LLMs + embedding models)
        """
        url = f"{self.base_url}/api/v1/models"

        try:
            r: Response = await self.http.get(url)
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ValueError(f"LM Studio API error {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise ConnectionError(f"Failed to reach LM Studio: {exc}") from exc

        data = r.json()
        return ModelsResponse.model_validate(data)
