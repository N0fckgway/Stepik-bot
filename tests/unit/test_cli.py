from typer.testing import CliRunner

from stepik_agent.cli import app


def test_dry_run_requires_explicit_cost_acknowledgement() -> None:
    result = CliRunner().invoke(app, ["dry-run", "https://stepik.org/lesson/1/step/1"])

    assert result.exit_code == 2
    assert "--allow-ai-cost" in result.output


def test_ai_smoke_requires_explicit_cost_acknowledgement() -> None:
    result = CliRunner().invoke(app, ["ai-smoke"])

    assert result.exit_code == 2
    assert "--allow-ai-cost" in result.output


def test_submit_one_requires_cost_and_write_acknowledgements() -> None:
    runner = CliRunner()

    without_cost = runner.invoke(
        app,
        ["submit-one", "https://stepik.org/lesson/1/step/1"],
    )
    without_submit = runner.invoke(
        app,
        [
            "submit-one",
            "https://stepik.org/lesson/1/step/1",
            "--allow-ai-cost",
        ],
    )

    assert without_cost.exit_code == 2
    assert "--allow-ai-cost" in without_cost.output
    assert without_submit.exit_code == 2
    assert "--confirm-submit" in without_submit.output


def test_resume_submit_requires_write_acknowledgement() -> None:
    result = CliRunner().invoke(app, ["resume-submit", "1"])

    assert result.exit_code == 2
    assert "--confirm-submit" in result.output


def test_run_requires_cost_and_write_acknowledgements() -> None:
    runner = CliRunner()

    without_cost = runner.invoke(app, ["run", "https://stepik.org/course/1780"])
    without_submit = runner.invoke(
        app,
        ["run", "https://stepik.org/course/1780", "--allow-ai-cost"],
    )

    assert without_cost.exit_code == 2
    assert "--allow-ai-cost" in without_cost.output
    assert without_submit.exit_code == 2
    assert "--confirm-submit" in without_submit.output


def test_run_exposes_explicit_local_resume_mode() -> None:
    result = CliRunner().invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    assert "--resume" in result.output
    assert "local course snapshot" in result.output


def test_run_lock_rejects_a_second_process(tmp_path) -> None:
    from stepik_agent.storage.run_lock import CourseRunAlreadyActiveError, CourseRunLock

    database_url = f"sqlite:///{tmp_path / 'agent.db'}"
    with CourseRunLock(database_url):
        try:
            with CourseRunLock(database_url):
                raise AssertionError("second lock unexpectedly acquired")
        except CourseRunAlreadyActiveError:
            pass


def test_doctor_reports_dedicated_reviewer_without_exposing_model(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANYMODEL_REVIEW_MODEL", "private-review-model")

    result = CliRunner().invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "AnyModel reviewer: yes" in result.output
    assert "private-review-model" not in result.output


def test_status_reports_missing_local_database_without_network(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["status"])

    assert result.exit_code == 0
    assert "database is not initialized" in result.output
