import stat

import pytest

pytest.importorskip("alembic")
pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine, inspect, text

from stepik_agent.storage.migrations import current_database_revision, upgrade_database


def test_upgrade_builds_latest_private_sqlite_schema(tmp_path) -> None:
    path = tmp_path / "migrated.db"
    database_url = f"sqlite:///{path}"

    upgrade_database(database_url)

    engine = create_engine(database_url)
    columns = {column["name"] for column in inspect(engine).get_columns("task_attempts")}
    unit_columns = {column["name"] for column in inspect(engine).get_columns("course_units")}
    indexes = {index["name"] for index in inspect(engine).get_indexes("task_attempts")}
    engine.dispose()

    assert {
        "state",
        "solution_language",
        "submission_reply",
        "attempt_dataset",
        "dataset_url",
    } <= columns
    assert "ix_task_attempts_state" in indexes
    assert "assignment_map" in unit_columns
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert current_database_revision(database_url) == "0005"


def test_upgrade_recovers_unversioned_initial_sqlite_database(tmp_path) -> None:
    path = tmp_path / "legacy.db"
    database_url = f"sqlite:///{path}"
    upgrade_database(database_url, revision="0001")
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM alembic_version"))
    engine.dispose()

    upgrade_database(database_url)

    engine = create_engine(database_url)
    with engine.connect() as connection:
        version = connection.scalar(text("SELECT version_num FROM alembic_version"))
    columns = {column["name"] for column in inspect(engine).get_columns("task_attempts")}
    unit_columns = {column["name"] for column in inspect(engine).get_columns("course_units")}
    engine.dispose()
    assert version == "0005"
    assert "assignment_map" in unit_columns
    assert {
        "state",
        "solution_language",
        "submission_reply",
        "attempt_dataset",
        "dataset_url",
    } <= columns
