"""Explicit Stepik write client with no automatic POST retries."""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from time import monotonic
from typing import Any

import httpx

from stepik_agent.core.errors import (
    AmbiguousStepikWriteError,
    StepikApiError,
    StepikAuthenticationError,
)

_PENDING_SUBMISSION_STATUS = "evaluation"


@dataclass(frozen=True, slots=True)
class StepikAttempt:
    attempt_id: int
    step_id: int
    dataset: dict[str, object] | str | None = None
    dataset_url: str | None = None


@dataclass(frozen=True, slots=True)
class StepikSubmission:
    submission_id: int
    attempt_id: int
    status: str
    hint: str | None = None

    @property
    def is_pending(self) -> bool:
        return self.status == _PENDING_SUBMISSION_STATUS

    @property
    def is_correct(self) -> bool:
        return self.status == "correct"


class StepikWriteClient:
    """Create one attempt/submission and poll it without replaying state-changing calls."""

    def __init__(
        self,
        *,
        access_token: str,
        base_url: str = "https://stepik.org",
        timeout_seconds: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not access_token.strip():
            raise ValueError("access_token must not be empty")
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {access_token}",
                "User-Agent": "stepik-autonomous-agent/0.1.0",
            },
            timeout=timeout_seconds,
            transport=transport,
        )

    async def __aenter__(self) -> "StepikWriteClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def create_attempt(self, step_id: int) -> StepikAttempt:
        _positive_id(step_id, field="step_id")
        item = await self._post_once(
            "/api/attempts",
            payload={"attempt": {"step": step_id}},
            collection="attempts",
        )
        try:
            attempt_id = _positive_id(item.get("id"), field="attempt.id")
            returned_step_id = _positive_id(item.get("step"), field="attempt.step")
        except StepikApiError as error:
            raise AmbiguousStepikWriteError(
                "Stepik created an attempt but returned an invalid identifier; retry is blocked"
            ) from error
        if returned_step_id != step_id:
            raise AmbiguousStepikWriteError(
                "Stepik created an attempt for a different step; retry is blocked"
            )
        dataset = item.get("dataset")
        if dataset is not None and not isinstance(dataset, (dict, str)):
            raise AmbiguousStepikWriteError(
                "Stepik created an attempt but returned an invalid dataset; retry is blocked"
            )
        dataset_url = item.get("dataset_url")
        if dataset_url is not None and not isinstance(dataset_url, str):
            raise AmbiguousStepikWriteError(
                "Stepik created an attempt but returned an invalid dataset URL; retry is blocked"
            )
        return StepikAttempt(
            attempt_id=attempt_id,
            step_id=step_id,
            dataset=dataset,
            dataset_url=dataset_url,
        )

    async def download_dataset(self, dataset_url: str, *, max_bytes: int = 2_000_000) -> str:
        if not dataset_url.startswith("/") or dataset_url.startswith("//"):
            raise StepikApiError("Attempt dataset URL must be a same-origin absolute path")
        try:
            response = await self._client.get(dataset_url)
        except httpx.RequestError as error:
            raise StepikApiError("Stepik dataset download failed") from error
        _raise_for_rejected_token(response)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise StepikApiError(
                f"Stepik dataset download returned HTTP {response.status_code}"
            ) from error
        if len(response.content) > max_bytes:
            raise StepikApiError("Stepik dataset exceeds the safe download size limit")
        try:
            return response.content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise StepikApiError("Stepik dataset is not UTF-8 text") from error

    async def record_view(self, *, step_id: int, assignment_id: int | None) -> None:
        _positive_id(step_id, field="step_id")
        if assignment_id is not None:
            _positive_id(assignment_id, field="assignment_id")
        try:
            response = await self._client.post(
                "/api/views",
                json={"view": {"assignment": assignment_id, "step": step_id}},
            )
        except httpx.RequestError as error:
            raise AmbiguousStepikWriteError(
                "Stepik view outcome is unknown after a network error; automatic retry is blocked"
            ) from error
        if response.status_code >= 500:
            raise AmbiguousStepikWriteError(
                f"Stepik view outcome is unknown after HTTP {response.status_code}; "
                "automatic retry is blocked"
            )
        _raise_for_rejected_token(response)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise StepikApiError(
                f"Stepik view report returned HTTP {response.status_code}"
            ) from error

    async def create_code_submission(
        self,
        *,
        attempt_id: int,
        code: str,
        language: str,
    ) -> StepikSubmission:
        _positive_id(attempt_id, field="attempt_id")
        if not code.strip() or len(code) > 100_000:
            raise ValueError("code must contain between 1 and 100000 characters")
        if not language.strip() or len(language) > 100:
            raise ValueError("language must contain between 1 and 100 characters")
        return await self.create_submission(
            attempt_id=attempt_id,
            reply={"code": code, "language": language},
        )

    async def create_submission(
        self,
        *,
        attempt_id: int,
        reply: dict[str, object],
    ) -> StepikSubmission:
        _positive_id(attempt_id, field="attempt_id")
        _validate_reply(reply)
        item = await self._post_once(
            "/api/submissions",
            payload={
                "submission": {
                    "attempt": attempt_id,
                    "reply": reply,
                    "time": _stepik_timestamp(),
                }
            },
            collection="submissions",
        )
        try:
            return _submission_from_item(item, expected_attempt_id=attempt_id)
        except AmbiguousStepikWriteError:
            raise
        except StepikApiError as error:
            raise AmbiguousStepikWriteError(
                "Stepik created a submission but returned an invalid object; retry is blocked"
            ) from error

    async def get_submission(self, submission_id: int) -> StepikSubmission:
        _positive_id(submission_id, field="submission_id")
        try:
            response = await self._client.get(f"/api/submissions/{submission_id}")
        except httpx.RequestError as error:
            raise StepikApiError("Stepik verdict request failed") from error
        _raise_for_rejected_token(response)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise StepikApiError(
                f"Stepik verdict request returned HTTP {response.status_code}"
            ) from error
        item = _single_item(response, collection="submissions", write_succeeded=False)
        submission = _submission_from_item(item)
        if submission.submission_id != submission_id:
            raise StepikApiError("Stepik returned a different submission than requested")
        return submission

    async def get_submissions_for_attempt(self, attempt_id: int) -> list[StepikSubmission]:
        """Read submissions for one attempt to reconcile an ambiguous POST safely."""

        _positive_id(attempt_id, field="attempt_id")
        try:
            response = await self._client.get(
                "/api/submissions",
                params={"attempt": attempt_id, "order": "desc"},
            )
        except httpx.RequestError as error:
            raise StepikApiError("Stepik submission reconciliation request failed") from error
        _raise_for_rejected_token(response)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise StepikApiError(
                f"Stepik submission reconciliation returned HTTP {response.status_code}"
            ) from error
        items = _collection_items(response, collection="submissions")
        submissions = [_submission_from_item(item) for item in items]
        if any(submission.attempt_id != attempt_id for submission in submissions):
            raise StepikApiError("Stepik returned a submission for a different attempt")
        return submissions

    async def wait_for_verdict(
        self,
        submission_id: int,
        *,
        timeout_seconds: float,
        initial_delay_seconds: float = 0.5,
        max_delay_seconds: float = 5.0,
    ) -> StepikSubmission:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if initial_delay_seconds <= 0 or max_delay_seconds <= 0:
            raise ValueError("poll delays must be positive")
        deadline = monotonic() + timeout_seconds
        delay = min(initial_delay_seconds, max_delay_seconds)
        while True:
            submission = await self.get_submission(submission_id)
            if not submission.is_pending:
                return submission
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise StepikApiError("Timed out waiting for the Stepik submission verdict")
            await asyncio.sleep(min(delay, remaining))
            delay = min(delay * 2, max_delay_seconds)

    async def _post_once(
        self,
        path: str,
        *,
        payload: dict[str, object],
        collection: str,
    ) -> dict[str, Any]:
        try:
            response = await self._client.post(path, json=payload)
        except httpx.RequestError as error:
            raise AmbiguousStepikWriteError(
                "Stepik write outcome is unknown after a network error; automatic retry is blocked"
            ) from error
        if response.status_code >= 500:
            raise AmbiguousStepikWriteError(
                f"Stepik write outcome is unknown after HTTP {response.status_code}; "
                "automatic retry is blocked"
            )
        _raise_for_rejected_token(response)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise StepikApiError(f"Stepik write returned HTTP {response.status_code}") from error
        return _single_item(response, collection=collection, write_succeeded=True)


def _raise_for_rejected_token(response: httpx.Response) -> None:
    if response.status_code == 401:
        raise StepikAuthenticationError(
            "Stepik rejected the OAuth token; run stepik-agent auth before retrying"
        )


def _single_item(
    response: httpx.Response,
    *,
    collection: str,
    write_succeeded: bool,
) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as error:
        if write_succeeded:
            raise AmbiguousStepikWriteError(
                "Stepik write succeeded at HTTP level but returned invalid JSON; retry is blocked"
            ) from error
        raise StepikApiError("Stepik returned invalid JSON") from error
    items = payload.get(collection) if isinstance(payload, dict) else None
    if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict):
        error_type = AmbiguousStepikWriteError if write_succeeded else StepikApiError
        raise error_type(f"Stepik response has no single '{collection}' object")
    return items[0]


def _collection_items(response: httpx.Response, *, collection: str) -> list[dict[str, Any]]:
    try:
        payload = response.json()
    except ValueError as error:
        raise StepikApiError("Stepik returned invalid JSON") from error
    items = payload.get(collection) if isinstance(payload, dict) else None
    if not isinstance(items, list) or len(items) > 1_000:
        raise StepikApiError(f"Stepik response has no valid '{collection}' collection")
    if not all(isinstance(item, dict) for item in items):
        raise StepikApiError(f"Stepik response contains an invalid '{collection}' object")
    return items


def _submission_from_item(
    item: dict[str, Any],
    *,
    expected_attempt_id: int | None = None,
) -> StepikSubmission:
    attempt_id = _positive_id(item.get("attempt"), field="submission.attempt")
    if expected_attempt_id is not None and attempt_id != expected_attempt_id:
        raise AmbiguousStepikWriteError(
            "Stepik returned a submission for a different attempt; retry is blocked"
        )
    status = item.get("status")
    if not isinstance(status, str) or not status.strip() or len(status) > 100:
        raise StepikApiError("submission.status must be a non-empty short string")
    hint = item.get("hint")
    if hint is not None and not isinstance(hint, str):
        raise StepikApiError("submission.hint must be a string or null")
    return StepikSubmission(
        submission_id=_positive_id(item.get("id"), field="submission.id"),
        attempt_id=attempt_id,
        status=status.strip(),
        hint=hint[:100_000] if hint is not None else None,
    )


def _positive_id(value: object, *, field: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    raise StepikApiError(f"{field} must be a positive integer")


def _validate_reply(reply: dict[str, object]) -> None:
    if not reply or len(reply) > 4:
        raise ValueError("reply must contain between 1 and 4 fields")
    allowed_fields = {
        "code",
        "language",
        "number",
        "text",
        "formula",
        "choices",
        "ordering",
        "blanks",
        "file",
    }
    if "code" in reply and set(reply) != {"code", "language"}:
        raise ValueError("code reply must contain exactly code and language")
    if "language" in reply and "code" not in reply:
        raise ValueError("language reply field requires code")
    if "code" in reply:
        for key in ("code", "language"):
            value = reply.get(key)
            if not isinstance(value, str) or not value.strip() or len(value) > 100_000:
                raise ValueError(f"reply {key} must contain between 1 and 100000 characters")
        return
    answer_fields = {
        "number",
        "text",
        "formula",
        "choices",
        "ordering",
        "blanks",
        "file",
    } & set(reply)
    if len(answer_fields) != 1:
        raise ValueError("reply must contain exactly one answer field")
    for key, value in reply.items():
        if key not in allowed_fields:
            raise ValueError(f"unsupported reply field: {key}")
        if isinstance(value, str):
            if not value.strip() or len(value) > 100_000:
                raise ValueError("reply text fields must contain between 1 and 100000 characters")
            if key == "number":
                try:
                    parsed = Decimal(value)
                except InvalidOperation as error:
                    raise ValueError("reply number must be a finite decimal") from error
                if not parsed.is_finite():
                    raise ValueError("reply number must be finite")
            continue
        if key == "choices":
            if not _valid_choice_reply(value):
                raise ValueError(
                    "reply choices must be a boolean list or a valid Stepik table choice list"
                )
            continue
        if key == "ordering":
            if (
                not isinstance(value, list)
                or not value
                or len(value) > 100
                or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
                or sorted(value) != list(range(len(value)))
            ):
                raise ValueError("reply ordering must be a zero-based permutation")
            continue
        if key == "blanks":
            if (
                not isinstance(value, list)
                or not value
                or len(value) > 100
                or not all(
                    isinstance(item, str) and item.strip() and len(item) <= 500 for item in value
                )
            ):
                raise ValueError("reply blanks must be a non-empty short string list")
            continue
        raise ValueError(f"unsupported reply value for field: {key}")


def _valid_choice_reply(value: object) -> bool:
    if not isinstance(value, list) or not value or len(value) > 100:
        return False
    if all(isinstance(item, bool) for item in value):
        return True
    for row in value:
        if not isinstance(row, dict) or set(row) != {"name_row", "columns"}:
            return False
        name_row = row.get("name_row")
        columns = row.get("columns")
        if not isinstance(name_row, str) or not name_row.strip() or len(name_row) > 500:
            return False
        if not isinstance(columns, list) or not columns or len(columns) > 100:
            return False
        for cell in columns:
            if not isinstance(cell, dict) or set(cell) != {"name", "answer"}:
                return False
            name = cell.get("name")
            answer = cell.get("answer")
            if not isinstance(name, str) or not name.strip() or len(name) > 500:
                return False
            if not isinstance(answer, bool):
                return False
    return True


def _stepik_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
