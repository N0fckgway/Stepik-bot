"""Process lock preventing concurrent course runs against one local database."""

from pathlib import Path
from types import TracebackType
from typing import TextIO
from urllib.parse import unquote

from sqlalchemy.engine import make_url

from stepik_agent.core.errors import StepikAgentError


class CourseRunAlreadyActiveError(StepikAgentError):
    """Raised when another autonomous course run already owns the lock."""


class CourseRunLock:
    """Use an advisory filesystem lock for the lifetime of one course run."""

    def __init__(self, database_url: str) -> None:
        self._path = _lock_path(database_url)
        self._handle: TextIO | None = None

    def __enter__(self) -> "CourseRunLock":
        import fcntl

        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._path.open("a+")
        self._handle = handle
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.close()
            self._handle = None
            raise CourseRunAlreadyActiveError(
                "another course run is already active; stop it before starting a new run"
            ) from error
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        import fcntl

        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


def _lock_path(database_url: str) -> Path:
    url = make_url(database_url)
    database_path = url.database
    if (
        url.get_backend_name() == "sqlite"
        and isinstance(database_path, str)
        and database_path != ":memory:"
    ):
        return Path(unquote(database_path)).expanduser().with_suffix(".run.lock")
    return Path("data/course.run.lock")
