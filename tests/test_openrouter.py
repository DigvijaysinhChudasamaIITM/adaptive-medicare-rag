import json

import httpx
import pytest

from app.clients.openrouter import (
    OpenRouterClient,
    OpenRouterConfigurationError,
    OpenRouterProviderError,
    OpenRouterResponseError,
)
from app.config import Settings
from app.models.generation import GroundingEvidence


def make_settings(
    *,
    api_key: str | None = "test-key",
) -> Settings:
    return Settings(
        openrouter_api_key=api_key,
        llm_model="primary-model",
        llm_fallback_model="fallback-model",
        llm_timeout_seconds=5,
    )


def success_response(
    *,
    model: str,
    answer: str = "Supported answer.",
    citations: list[str] | None = None,
) -> httpx.Response:
    if citations is None:
        citations = ["chunk-1"]

    return httpx.Response(
        200,
        json={
            "model": model,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "answer": answer,
                                "citations": citations,
                            }
                        )
                    },
                }
            ],
        },
    )


def evidence() -> list[GroundingEvidence]:
    return [
        GroundingEvidence(
            chunk_id="chunk-1",
            text="Relevant Medicare evidence.",
        )
    ]


@pytest.mark.asyncio
async def test_openrouter_successfully_validates_generation() -> None:
    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        payload = json.loads(request.content)

        assert payload["model"] == "primary-model"
        assert payload["temperature"] == 0
        assert payload["reasoning"] == {"effort": "none"}
        assert (
            payload["response_format"]["type"]
            == "json_schema"
        )

        return success_response(model="primary-model")

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport
    ) as http_client:
        client = OpenRouterClient(
            make_settings(),
            http_client=http_client,
        )

        result = await client.generate(
            question="Question?",
            evidence=evidence(),
        )

    assert result.output.answer == "Supported answer."
    assert result.output.citations == ["chunk-1"]
    assert result.requested_model == "primary-model"
    assert result.returned_model == "primary-model"
    assert result.used_fallback is False


@pytest.mark.asyncio
async def test_transient_error_is_retried() -> None:
    calls = 0

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal calls
        calls += 1

        if calls == 1:
            return httpx.Response(
                500,
                json={"error": {"message": "temporary"}},
            )

        return success_response(model="primary-model")

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport
    ) as http_client:
        client = OpenRouterClient(
            make_settings(),
            http_client=http_client,
            max_attempts_per_model=2,
        )

        result = await client.generate(
            question="Question?",
            evidence=evidence(),
        )

    assert calls == 2
    assert result.used_fallback is False


@pytest.mark.asyncio
async def test_unavailable_primary_uses_fallback() -> None:
    requested_models: list[str] = []

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        payload = json.loads(request.content)
        model = payload["model"]
        requested_models.append(model)

        if model == "primary-model":
            return httpx.Response(
                404,
                json={"error": {"message": "unavailable"}},
            )

        return success_response(model="actual-fallback-model")

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport
    ) as http_client:
        client = OpenRouterClient(
            make_settings(),
            http_client=http_client,
        )

        result = await client.generate(
            question="Question?",
            evidence=evidence(),
        )

    assert requested_models == [
        "primary-model",
        "fallback-model",
    ]
    assert result.requested_model == "fallback-model"
    assert result.returned_model == "actual-fallback-model"
    assert result.used_fallback is True


@pytest.mark.asyncio
async def test_timeout_retries_then_uses_fallback() -> None:
    primary_calls = 0

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal primary_calls

        payload = json.loads(request.content)
        model = payload["model"]

        if model == "primary-model":
            primary_calls += 1
            raise httpx.ReadTimeout(
                "timed out",
                request=request,
            )

        return success_response(model="fallback-model")

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport
    ) as http_client:
        client = OpenRouterClient(
            make_settings(),
            http_client=http_client,
            max_attempts_per_model=2,
        )

        result = await client.generate(
            question="Question?",
            evidence=evidence(),
        )

    assert primary_calls == 2
    assert result.used_fallback is True


@pytest.mark.asyncio
async def test_authentication_failure_does_not_retry_or_fallback() -> None:
    calls = 0

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal calls
        calls += 1

        return httpx.Response(
            401,
            json={"error": {"message": "invalid key"}},
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport
    ) as http_client:
        client = OpenRouterClient(
            make_settings(),
            http_client=http_client,
        )

        with pytest.raises(OpenRouterProviderError) as exc_info:
            await client.generate(
                question="Question?",
                evidence=evidence(),
            )

    assert calls == 1
    assert exc_info.value.status_code == 401
    assert exc_info.value.retryable is False
    assert exc_info.value.fallback_allowed is False


@pytest.mark.asyncio
async def test_malformed_primary_output_can_fall_back() -> None:
    requested_models: list[str] = []

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        payload = json.loads(request.content)
        model = payload["model"]
        requested_models.append(model)

        if model == "primary-model":
            return httpx.Response(
                200,
                json={
                    "model": model,
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "{not valid json"
                            },
                        }
                    ],
                },
            )

        return success_response(model="fallback-model")

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport
    ) as http_client:
        client = OpenRouterClient(
            make_settings(),
            http_client=http_client,
        )

        result = await client.generate(
            question="Question?",
            evidence=evidence(),
        )

    assert requested_models == [
        "primary-model",
        "fallback-model",
    ]
    assert result.used_fallback is True


@pytest.mark.asyncio
async def test_malformed_output_from_all_models_fails_safely() -> None:
    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "bad-model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "{bad-json"
                        },
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport
    ) as http_client:
        client = OpenRouterClient(
            make_settings(),
            http_client=http_client,
        )

        with pytest.raises(OpenRouterResponseError):
            await client.generate(
                question="Question?",
                evidence=evidence(),
            )


@pytest.mark.asyncio
async def test_missing_api_key_fails_before_network_call() -> None:
    calls = 0

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal calls
        calls += 1
        return success_response(model="primary-model")

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport
    ) as http_client:
        client = OpenRouterClient(
            make_settings(api_key=None),
            http_client=http_client,
        )

        with pytest.raises(OpenRouterConfigurationError):
            await client.generate(
                question="Question?",
                evidence=evidence(),
            )

    assert calls == 0

@pytest.mark.asyncio
async def test_all_models_failing_raises_safe_provider_error() -> None:
    requested_models: list[str] = []

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        payload = json.loads(request.content)
        requested_models.append(payload["model"])

        return httpx.Response(
            503,
            json={
                "error": {
                    "message": "temporary provider failure",
                }
            },
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport
    ) as http_client:
        client = OpenRouterClient(
            make_settings(),
            http_client=http_client,
            max_attempts_per_model=2,
        )

        with pytest.raises(OpenRouterProviderError) as exc_info:
            await client.generate(
                question="Question?",
                evidence=evidence(),
            )

    assert requested_models == [
        "primary-model",
        "primary-model",
        "fallback-model",
        "fallback-model",
    ]

    assert exc_info.value.status_code == 503
    assert exc_info.value.retryable is True
    assert exc_info.value.fallback_allowed is True