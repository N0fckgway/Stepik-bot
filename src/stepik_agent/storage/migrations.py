from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

from stepik_agent.storage.database import secure_sqlite_database_file


def upgrade_database(database_url: str, *, revision: str = "head") -> None:
    """Apply repository migrations to the configured database."""

    project_root = Path(__file__).resolve().parents[3]
    config_path = project_root / "alembic.ini"
    if not config_path.is_file():
        raise RuntimeError(f"Alembic configuration not found at {config_path}")
    config = Config(config_path)
    config.set_main_option("script_location", str(project_root / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    _repair_unversioned_initial_sqlite(config, database_url)
    command.upgrade(config, revision)
    secure_sqlite_database_file(database_url)


def current_database_revision(database_url: str) -> str | None:
    url = make_url(database_url)
    database_path = url.database
    if url.get_backend_name() == "sqlite" and database_path not in {None, ":memory:"}:
        assert database_path is not None
        if not Path(database_path).expanduser().exists():
            return None
    engine = create_engine(url)
    try:
        if "alembic_version" not in inspect(engine).get_table_names():
            return None
        with engine.connect() as connection:
            versions = list(connection.scalars(text("SELECT version_num FROM alembic_version")))
    finally:
        engine.dispose()
    if not versions:
        return None
    if len(versions) != 1:
        raise RuntimeError("Database contains multiple Alembic revisions")
    version = versions[0]
    if not isinstance(version, str) or not version:
        raise RuntimeError("Database contains an invalid Alembic revision")
    return version


def _repair_unversioned_initial_sqlite(config: Config, database_url: str) -> None:
    """Recover databases created before SQLite revision commits were fixed."""

    url = make_url(database_url)
    database_path = url.database
    if url.get_backend_name() != "sqlite" or database_path is None or database_path == ":memory:":
        return
    path = Path(database_path).expanduser()
    if not path.exists():
        return

    engine = create_engine(url)
    try:
        schema = inspect(engine)
        tables = set(schema.get_table_names())
        initial_tables = {
            "courses",
            "lessons",
            "course_units",
            "tasks",
            "runs",
            "task_attempts",
            "ai_requests",
            "event_logs",
        }
        if "alembic_version" not in tables:
            if tables:
                raise RuntimeError("Existing SQLite schema has no Alembic revision table")
            return
        with engine.connect() as connection:
            versions = list(connection.scalars(text("SELECT version_num FROM alembic_version")))
        if versions:
            return
        if not initial_tables <= tables:
            raise RuntimeError("Unversioned SQLite schema does not match the initial migration")
        attempt_columns = {column["name"] for column in schema.get_columns("task_attempts")}
        unit_columns = {column["name"] for column in schema.get_columns("course_units")}
        recovered_revision = (
            "0005"
            if "assignment_map" in unit_columns
            and {
                "state",
                "solution_language",
                "submission_reply",
                "attempt_dataset",
                "dataset_url",
            }
            <= attempt_columns
            else "0004"
            if {
                "state",
                "solution_language",
                "submission_reply",
                "attempt_dataset",
                "dataset_url",
            }
            <= attempt_columns
            else "0003"
            if {"state", "solution_language", "submission_reply"} <= attempt_columns
            else "0002"
            if {"state", "solution_language"} <= attempt_columns
            else "0001"
        )
    finally:
        engine.dispose()

    command.stamp(config, recovered_revision)
