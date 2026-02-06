from typing import Iterable, Optional

from pydantic import ByteSize

import api_models
import lms_rest_api

# import random
# import string
# def random_sha() -> str:
#     return "".join(random.choices(string.hexdigits.lower(), k=64))


async def list_lms_models(
    lms_api_client: lms_rest_api.LMStudioRestAPIClient,
    *,
    loaded_only: bool,
    specific_model: Optional[str] = None,
) -> api_models.ListResponse:
    lms_all_models = await lms_api_client.list_models()
    models: Iterable[lms_rest_api.ModelInfo]
    models = lms_all_models.models
    if loaded_only:
        models = filter(lambda m: len(m.loaded_instances) > 0, models)
    if specific_model is not None:
        models = filter(lambda m: m.key == specific_model, models)

    ollama_models = [
        api_models.ListResponse.Model(
            name=model.key,
            model=model.key,
            size=ByteSize(model.size_bytes),
            details=api_models.ModelDetails(
                format=model.format or "",
                family=model.architecture or "",
                families=model.architecture and [model.architecture] or [""],
                parameter_size=model.params_string or "",
                quantization_level=(
                    (model.quantization is not None) and model.quantization.name
                )
                or "",
            ),
        )
        for model in models
    ]

    return api_models.ListResponse(models=ollama_models)
