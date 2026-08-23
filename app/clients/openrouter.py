import json
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.models.generation import GeneratedAnswer, GroundingEvidence
from app.rag.prompting import build_grounded_messages

OPENROUTER_CHAT_COMPLETIONS_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)

TRANSIENT_STATUS_CODES = {
    408,
    429,
}


class OpenRouterError(RuntimeError):
    """Base error for OpenRouter generation failures."""


class OpenRouterConfigurationError(OpenRouterError):
    """Raised when required OpenRouter configuration is missing."""


class OpenRouterProviderError(OpenRouterError):
    """Raised when OpenRouter or an upstream model provider fails."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        fallback_allowed: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.fallback_allowed = fallback_allowed


class OpenRouterResponseError(OpenRouterError):
    """Raised when a successful provider response is unusable."""


@dataclass(frozen=True, slots=True)
class OpenRouterGenerationResult:
    """Validated generation plus non-authoritative provider metadata."""

    output: GeneratedAnswer
    requested_model: str
    returned_model: str | None
    used_fallback: bool


class OpenRouterClient:
    """Minimal asynchronous OpenRouter chat-completions client."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        http_client: httpx.AsyncClient | None = None,
        max_attempts_per_model: int = 2,
        max_tokens: int = 512,
    ) -> None:
        self._settings = settings or get_settings()

        if max_attempts_per_model < 1:
            raise ValueError("max_attempts_per_model must be at least 1.")

        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1.")

        self._max_attempts_per_model = max_attempts_per_model
        self._max_tokens = max_tokens

        self._owns_http_client = http_client is None

        if http_client is None:
            timeout_seconds = self._settings.llm_timeout_seconds
            short_timeout = min(10.0, timeout_seconds)

            timeout = httpx.Timeout(
                connect=short_timeout,
                read=timeout_seconds,
                write=short_timeout,
                pool=short_timeout,
            )

            http_client = httpx.AsyncClient(timeout=timeout)

        self._http_client = http_client

    async def aclose(self) -> None:
        """Close the internally owned HTTP client."""

        if self._owns_http_client:
            await self._http_client.aclose()

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        await self.aclose()

    async def generate(
        self,
        *,
        question: str,
        evidence: list[GroundingEvidence],
    ) -> OpenRouterGenerationResult:
        """Generate a grounded answer with bounded retry and fallback."""

        api_key = self._settings.openrouter_api_key
        primary_model = self._settings.llm_model
        fallback_model = self._settings.llm_fallback_model

        if not api_key:
            raise OpenRouterConfigurationError(
                "OPENROUTER_API_KEY is not configured."
            )

        if not primary_model:
            raise OpenRouterConfigurationError(
                "LLM_MODEL is not configured."
            )

        models = [primary_model]

        if (
            fallback_model
            and fallback_model != primary_model
        ):
            models.append(fallback_model)

        messages = build_grounded_messages(
            question=question,
            evidence=evidence,
        )

        last_error: OpenRouterError | None = None

        for model_index, model in enumerate(models):
            used_fallback = model_index > 0

            try:
                return await self._generate_with_model(
                    api_key=api_key,
                    model=model,
                    messages=messages,
                    used_fallback=used_fallback,
                )
            except OpenRouterProviderError as exc:
                last_error = exc

                has_fallback = model_index < len(models) - 1

                if not exc.fallback_allowed or not has_fallback:
                    raise
            except OpenRouterResponseError as exc:
                last_error = exc

                has_fallback = model_index < len(models) - 1

                if not has_fallback:
                    raise

        if last_error is not None:
            raise last_error

        raise OpenRouterError("Generation failed without a recorded error.")

    async def _generate_with_model(
        self,
        *,
        api_key: str,
        model: str,
        messages: list[dict[str, str]],
        used_fallback: bool,
    ) -> OpenRouterGenerationResult:
        payload = self._build_payload(
            model=model,
            messages=messages,
        )

        for attempt in range(1, self._max_attempts_per_model + 1):
            try:
                response = await self._http_client.post(
                    OPENROUTER_CHAT_COMPLETIONS_URL,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
            except (
                httpx.TimeoutException,
                httpx.TransportError,
            ) as exc:
                if attempt < self._max_attempts_per_model:
                    continue

                raise OpenRouterProviderError(
                    "OpenRouter request failed due to a network or timeout "
                    "error.",
                    retryable=True,
                    fallback_allowed=True,
                ) from exc

            if response.is_success:
                return self._parse_success_response(
                    response=response,
                    requested_model=model,
                    used_fallback=used_fallback,
                )

            status_code = response.status_code
            retryable = self._is_transient_status(status_code)

            if (
                retryable
                and attempt < self._max_attempts_per_model
            ):
                continue

            fallback_allowed = (
                retryable
                or status_code == 404
            )

            raise OpenRouterProviderError(
                self._safe_provider_error_message(status_code),
                status_code=status_code,
                retryable=retryable,
                fallback_allowed=fallback_allowed,
            )

        raise OpenRouterProviderError(
            "OpenRouter request attempts were exhausted.",
            retryable=True,
            fallback_allowed=True,
        )

    def _build_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        return {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self._max_tokens,
            "reasoning": {
                "effort": "none",
            },
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "generated_answer",
                    "strict": True,
                    "schema": GeneratedAnswer.model_json_schema(),
                },
            },
            "provider": {
                "require_parameters": True,
            },
        }

    @staticmethod
    def _parse_success_response(
        *,
        response: httpx.Response,
        requested_model: str,
        used_fallback: bool,
    ) -> OpenRouterGenerationResult:
        try:
            body = response.json()
        except ValueError as exc:
            raise OpenRouterResponseError(
                "OpenRouter returned a non-JSON response."
            ) from exc

        try:
            choice = body["choices"][0]
            message = choice["message"]
            content = message["content"]
        except (
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            raise OpenRouterResponseError(
                "OpenRouter response is missing the expected completion "
                "content."
            ) from exc

        if choice.get("finish_reason") == "length":
            raise OpenRouterResponseError(
                "OpenRouter completion ended because the token limit "
                "was reached."
            )

        if not isinstance(content, str):
            raise OpenRouterResponseError(
                "OpenRouter completion content is not a string."
            )

        try:
            parsed_content = json.loads(content)
        except json.JSONDecodeError as exc:
            raise OpenRouterResponseError(
                "OpenRouter completion content is not valid JSON."
            ) from exc

        try:
            output = GeneratedAnswer.model_validate(parsed_content)
        except ValidationError as exc:
            raise OpenRouterResponseError(
                "OpenRouter completion does not match the generation schema."
            ) from exc

        returned_model = body.get("model")

        if returned_model is not None:
            returned_model = str(returned_model)

        return OpenRouterGenerationResult(
            output=output,
            requested_model=requested_model,
            returned_model=returned_model,
            used_fallback=used_fallback,
        )

    @staticmethod
    def _is_transient_status(status_code: int) -> bool:
        return (
            status_code in TRANSIENT_STATUS_CODES
            or 500 <= status_code <= 599
        )

    @staticmethod
    def _safe_provider_error_message(status_code: int) -> str:
        if status_code in {401, 403}:
            return "OpenRouter authentication or authorization failed."

        if status_code == 404:
            return "The requested OpenRouter model is unavailable."

        if status_code == 429:
            return "OpenRouter rate limit was reached."

        if 500 <= status_code <= 599:
            return "OpenRouter or its upstream provider is unavailable."

        return (
            "OpenRouter rejected the generation request "
            f"with HTTP {status_code}."
        )