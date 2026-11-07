import asyncio
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import httpx

from stepik_agent.core.errors import StepikApiError

_RESOURCE_RE = re.compile(r"^[a-z][a-z0-9-]*$")
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


class StepikReadClient:
    """Small read-only Stepik client used during discovery and synchronization."""

    def __init__(
        self,
        *,
        base_url: str = "https://stepik.org",
        access_token: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {
            "Accept": "application/json",
            "User-Agent": "stepik-autonomous-agent/0.1.0",
        }
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout_seconds,
            transport=transport,
        )
        self._max_retries = max_retries

    async def __aenter__(self) -> "StepikReadClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get_by_id(self, resource: str, object_id: int) -> dict[str, Any]:
        if object_id <= 0:
            raise ValueError("object_id must be positive")
        objects, _ = await self.get_page(resource, path_suffix=str(object_id))
        if not objects:
            raise StepikApiError(f"Stepik returned no {resource} object for id {object_id}")
        return objects[0]

    async def get_many(
        self,
        resource: str,
        object_ids: Sequence[int | str],
        *,
        batch_size: int = 50,
    ) -> list[dict[str, Any]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not object_ids:
            return []

        objects: list[dict[str, Any]] = []
        for offset in range(0, len(object_ids), batch_size):
            batch = object_ids[offset : offset + batch_size]
            for object_id in batch:
                if isinstance(object_id, bool):
                    raise ValueError("object ids must be positive integers or non-empty strings")
                if isinstance(object_id, int) and object_id <= 0:
                    raise ValueError("integer object ids must be positive")
                if isinstance(object_id, str) and not object_id.strip():
                    raise ValueError("string object ids must not be empty")
            page = 1
            while True:
                params = [("ids[]", str(object_id)) for object_id in batch]
                params.append(("page", str(page)))
                page_objects, meta = await self.get_page(resource, params=params)
                objects.extend(page_objects)
                if not meta.get("has_next", False):
                    break
                page += 1
        return objects

    async def iter_all(
        self,
        resource: str,
        *,
        params: Mapping[str, str | int] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        page = 1
        while True:
            page_params: dict[str, str | int] = dict(params or {})
            page_params["page"] = page
            objects, meta = await self.get_page(resource, params=page_params)
            for item in objects:
                yield item
            if not meta.get("has_next", False):
                return
            page += 1

    async def get_page(
        self,
        resource: str,
        *,
        path_suffix: str | None = None,
        params: Mapping[str, str | int] | list[tuple[str, str]] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        self._validate_resource(resource)
        path = f"/api/{resource}"
        if path_suffix is not None:
            path = f"{path}/{path_suffix}"
        response = await self._get_with_retry(path, params=params)
        try:
            payload = response.json()
        except ValueError as error:
            raise StepikApiError("Stepik returned a non-JSON response") from error
        if not isinstance(payload, dict):
            raise StepikApiError("Stepik response must be a JSON object")
        objects = payload.get(resource)
        if not isinstance(objects, list) or not all(isinstance(item, dict) for item in objects):
            raise StepikApiError(f"Stepik response has no valid '{resource}' collection")
        meta = payload.get("meta", {})
        if not isinstance(meta, dict):
            raise StepikApiError("Stepik response has invalid pagination metadata")
        return objects, meta

    async def _get_with_retry(
        self,
        path: str,
        *,
        params: Mapping[str, str | int] | list[tuple[str, str]] | None,
    ) -> httpx.Response:
        response: httpx.Response | None = None
        query_params = _query_params(params)
        for retry in range(self._max_retries + 1):
            try:
                response = await self._client.get(path, params=query_params)
            except httpx.RequestError as error:
                if retry >= self._max_retries:
                    raise StepikApiError(f"Stepik request failed: {error}") from error
                await asyncio.sleep(2**retry)
                continue
            if response.status_code not in _RETRYABLE_STATUSES or retry >= self._max_retries:
                break
            await asyncio.sleep(self._retry_delay(response, retry))
        if response is None:
            raise StepikApiError("Stepik request did not produce a response")
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise StepikApiError(f"Stepik returned HTTP {response.status_code}") from error
        return response

    @staticmethod
    def _retry_delay(response: httpx.Response, retry: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            return min(float(retry_after), 60.0)
        return min(float(2**retry), 60.0)

    @staticmethod
    def _validate_resource(resource: str) -> None:
        if not _RESOURCE_RE.fullmatch(resource):
            raise ValueError("invalid Stepik API resource name")


def _query_params(
    params: Mapping[str, str | int] | list[tuple[str, str]] | None,
) -> httpx.QueryParams | None:
    if params is None:
        return None
    if isinstance(params, list):
        pairs: list[tuple[str, str | int | float | bool | None]] = [
            (key, value) for key, value in params
        ]
        return httpx.QueryParams(pairs)
    values: dict[
        str, str | int | float | bool | Sequence[str | int | float | bool | None] | None
    ] = {key: value for key, value in params.items()}
    return httpx.QueryParams(values)
