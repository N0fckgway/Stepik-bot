from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from stepik_agent.storage.models import Base


def secure_sqlite_database_file(database_url: str) -> None:
    path = _sqlite_database_path(database_url)
    if path is None or not path.exists():
        return
    if path.is_symlink():
        raise ValueError("SQLite database path must not be a symbolic link")
    path.chmod(0o600)


def _sqlite_database_path(database_url: str) -> Path | None:
    url = make_url(database_url)
    database_path = url.database
    if url.get_backend_name() != "sqlite" or database_path is None or database_path == ":memory:":
        return None
    return Path(database_path).expanduser()


class Database:
    """Own the SQLAlchemy engine and transaction boundary."""

    def __init__(self, database_url: str) -> None:
        url = make_url(database_url)
        database_path = _sqlite_database_path(database_url)
        if database_path is not None:
            database_path.parent.mkdir(parents=True, exist_ok=True)
            secure_sqlite_database_file(database_url)

        connect_args = {"check_same_thread": False} if url.get_backend_name() == "sqlite" else {}
        self.engine: Engine = create_engine(
            url,
            connect_args=connect_args,
            future=True,
            pool_pre_ping=True,
        )
        if url.get_backend_name() == "sqlite":
            self._configure_sqlite(self.engine, database_path)
        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=Session,
            expire_on_commit=False,
        )

    @staticmethod
    def _configure_sqlite(engine: Engine, database_path: Path | None) -> None:
        @event.listens_for(engine, "connect")
        def set_sqlite_pragmas(dbapi_connection: object, _connection_record: object) -> None:
            if database_path is not None:
                if database_path.is_symlink():
                    raise ValueError("SQLite database path must not be a symbolic link")
                database_path.chmod(0o600)
            cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA busy_timeout=5000")
            finally:
                cursor.close()

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        self.engine.dispose()
