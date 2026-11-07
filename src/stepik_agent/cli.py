import asyncio
import secrets
import webbrowser
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from sqlalchemy.exc import SQLAlchemyError

from stepik_agent import __version__
from stepik_agent.ai.budget import SharedAIBudget
from stepik_agent.ai.factory import create_anymodel_gateway
from stepik_agent.ai.gateway import AIRequestEvent
from stepik_agent.application.ai_smoke import run_ai_smoke
from stepik_agent.application.code_task import CodeTaskResult, CodeTaskRunner
from stepik_agent.application.course_run import CourseRunner, CourseRunResult
from stepik_agent.application.dry_run import DryRunResult, DryRunService
from stepik_agent.application.single_submit import SingleTaskSubmitService
from stepik_agent.config import Settings
from stepik_agent.core.errors import ConfigurationError, InvalidStepikUrlError, StepikAgentError
from stepik_agent.core.states import EventSeverity, RunState, TaskStatus
from stepik_agent.reporting.dry_run import DryRunReportStore
from stepik_agent.stepik.auth import StepikOAuthClient, TokenStore, wait_for_callback
from stepik_agent.stepik.client import StepikReadClient
from stepik_agent.stepik.sync import CourseSyncService, SyncResult
from stepik_agent.stepik.task_loader import StepikTaskLoader
from stepik_agent.stepik.urls import parse_stepik_url
from stepik_agent.stepik.write_client import StepikWriteClient
from stepik_agent.storage.attempts import AttemptRecord, AttemptStore
from stepik_agent.storage.course_runs import CourseRunStore
from stepik_agent.storage.database import Database
from stepik_agent.storage.migrations import current_database_revision, upgrade_database
from stepik_agent.storage.repositories import CatalogRepository, EventRepository
from stepik_agent.storage.run_lock import CourseRunLock
from stepik_agent.tasks.analyzer import TaskAnalyzer

app = typer.Typer(
    name="stepik-agent",
    help="Controlled local agent for supported Stepik coursework.",
    no_args_is_help=True,
)

_DEFAULT_DRY_RUN_OUTPUT = Path("data/dry_runs")
_DEFAULT_AI_SMOKE_OUTPUT = Path("data/ai_smoke")


@app.command("version")
def version() -> None:
    """Print the application version."""

    typer.echo(__version__)


@app.command("inspect-url")
def inspect_url(url: str) -> None:
    """Parse a Stepik URL without making a network request."""

    try:
        target = parse_stepik_url(url)
    except InvalidStepikUrlError as error:
        typer.echo(f"Invalid Stepik URL: {error}", err=True)
        raise typer.Exit(code=2) from error
    for key, value in asdict(target).items():
        typer.echo(f"{key}: {value}")


@app.command("doctor")
def doctor() -> None:
    """Check local configuration without exposing secret values."""

    try:
        settings = Settings()
    except StepikAgentError as error:
        typer.echo(f"Configuration error: {error}", err=True)
        raise typer.Exit(code=2) from error

    checks = {
        "Stepik OAuth": settings.stepik_oauth_configured,
        "Stepik token": TokenStore(settings.stepik_token_path).exists,
        "AnyModel": settings.ai_configured,
        "AnyModel reviewer": bool(settings.anymodel_review_model),
        "Dry run": settings.dry_run,
        "Auto submit": settings.auto_submit,
    }
    for label, enabled in checks.items():
        typer.echo(f"{label}: {'yes' if enabled else 'no'}")

    if not settings.dry_run and settings.auto_submit:
        typer.echo("WARNING: submissions are enabled")
    else:
        typer.echo("Submission mode: safe (no automatic submissions)")


@app.command("auth")
def auth(
    no_browser: bool = typer.Option(
        False,
        "--no-browser",
        help="Print the authorization URL without opening the default browser.",
    ),
) -> None:
    """Authorize the application using Stepik OAuth 2 Authorization Code flow."""

    try:
        settings = Settings()
        if not settings.stepik_oauth_configured:
            raise ConfigurationError(
                "STEPIK_CLIENT_ID and STEPIK_CLIENT_SECRET must be configured in .env"
            )
        assert settings.stepik_client_id is not None
        assert settings.stepik_client_secret is not None

        redirect_uri = str(settings.stepik_redirect_uri)
        state = secrets.token_urlsafe(32)
        oauth_client = StepikOAuthClient(
            base_url=str(settings.stepik_base_url),
            client_id=settings.stepik_client_id.get_secret_value(),
            client_secret=settings.stepik_client_secret.get_secret_value(),
        )
        authorization_url = oauth_client.authorization_url(
            redirect_uri=redirect_uri,
            state=state,
        )
        typer.echo("Open this URL to authorize Stepik Agent:")
        typer.echo(authorization_url)
        if not no_browser and not webbrowser.open(authorization_url):
            typer.echo("The browser could not be opened automatically; open the URL manually.")

        code = wait_for_callback(
            redirect_uri=redirect_uri,
            expected_state=state,
            timeout_seconds=settings.oauth_callback_timeout_seconds,
        )
        token = oauth_client.exchange_code(code=code, redirect_uri=redirect_uri)
        TokenStore(settings.stepik_token_path).save(token)
    except StepikAgentError as error:
        typer.echo(f"Authorization error: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(f"Stepik authorization saved to {settings.stepik_token_path}")


@app.command("db-upgrade")
def db_upgrade() -> None:
    """Apply all pending database migrations."""

    try:
        settings = Settings()
        upgrade_database(settings.database_url)
    except Exception as error:
        typer.echo(f"Database migration failed: {type(error).__name__}", err=True)
        raise typer.Exit(code=2) from error
    typer.echo("Database is at the latest migration")


@app.command("sync")
def sync_course(url: str) -> None:
    """Synchronize course structure and progress without submitting answers."""

    database: Database | None = None
    try:
        settings = Settings()
        target = parse_stepik_url(url)
        token = TokenStore(settings.stepik_token_path).load()
        active_database = Database(settings.database_url)
        database = active_database

        async def run_sync() -> SyncResult:
            async with StepikReadClient(
                base_url=str(settings.stepik_base_url),
                access_token=token.access_token,
            ) as client:
                with active_database.session() as session:
                    service = CourseSyncService(
                        client=client,
                        writer=CatalogRepository(session),
                    )
                    return await service.sync(target)

        result = asyncio.run(run_sync())
    except (StepikAgentError, SQLAlchemyError, ValueError) as error:
        typer.echo(f"Synchronization failed: {error}", err=True)
        raise typer.Exit(code=2) from error
    finally:
        if database is not None:
            database.dispose()

    typer.echo(
        "Synchronized "
        f"course={result.course_id} sections={result.sections} units={result.units} "
        f"lessons={result.lessons} tasks={result.tasks} passed={result.passed_tasks}"
    )
    if result.missing_assignment_ids:
        typer.echo(
            "WARNING: Stepik did not return optional assignment metadata for ids: "
            + ", ".join(str(value) for value in result.missing_assignment_ids),
            err=True,
        )


@app.command("status")
def status() -> None:
    """Show the latest local course run without contacting Stepik."""

    database: Database | None = None
    try:
        settings = Settings()
        active_database = Database(settings.database_url)
        database = active_database
        from sqlalchemy import inspect

        if not inspect(active_database.engine).has_table("runs"):
            typer.echo("Status: database is not initialized; run stepik-agent db-upgrade")
            return

        summary = CourseRunStore(active_database).latest_status()
    except (StepikAgentError, SQLAlchemyError, OSError, ValueError) as error:
        typer.echo(f"Status failed: {error}", err=True)
        raise typer.Exit(code=2) from error
    finally:
        if database is not None:
            database.dispose()

    if summary is None:
        typer.echo("Status: no local course runs found")
        return

    def format_time(value: datetime | None) -> str:
        return value.isoformat(sep=" ", timespec="seconds") if value is not None else "-"

    typer.echo(f"Course: {summary.course_title} (id={summary.course_id})")
    typer.echo(f"Course URL: {summary.course_url}")
    typer.echo(f"Run: {summary.run_id}")
    typer.echo(f"State: {summary.state.value}")
    typer.echo(f"Started: {format_time(summary.start_time)}")
    typer.echo(f"Finished: {format_time(summary.end_time)}")
    typer.echo(
        "Progress: "
        f"completed={summary.completed_tasks} failed={summary.failed_tasks} "
        f"total={summary.total_tasks}"
    )
    typer.echo(f"Estimated cost: ${summary.estimated_cost}")
    if summary.task_counts:
        counts = ", ".join(
            f"{status_name}={count}" for status_name, count in sorted(summary.task_counts.items())
        )
        typer.echo(f"Tasks by status: {counts}")
    if summary.state is RunState.COMPLETED:
        typer.echo("Completion: confirmed by the final course check")
    elif summary.state is RunState.RUNNING:
        typer.echo("Completion: run is still marked as running")
    else:
        typer.echo("Completion: not confirmed; inspect unsupported/failed/blocked tasks")


@app.command("run")
def run_course(
    url: str,
    allow_ai_cost: bool = typer.Option(
        False,
        "--allow-ai-cost",
        help="Allow bounded, potentially billable AnyModel calls for the run.",
    ),
    confirm_submit: bool = typer.Option(
        False,
        "--confirm-submit",
        help="Confirm autonomous Stepik attempts and submissions for this run.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from the latest local course snapshot without an initial full catalog sync.",
    ),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for per-task JSON reports."),
    ] = _DEFAULT_DRY_RUN_OUTPUT,
) -> None:
    """Synchronize and execute the supported tasks in one Stepik course."""

    if not allow_ai_cost:
        typer.echo(
            "Run stopped before AI call: pass --allow-ai-cost to acknowledge possible cost.",
            err=True,
        )
        raise typer.Exit(code=2)
    if not confirm_submit:
        typer.echo(
            "Run stopped before network writes: pass --confirm-submit for autonomous submissions.",
            err=True,
        )
        raise typer.Exit(code=2)

    database: Database | None = None
    events: list[AIRequestEvent] = []
    try:
        settings = Settings()
        if settings.dry_run or not settings.auto_submit:
            raise ConfigurationError("run requires DRY_RUN=false and AUTO_SUBMIT=true")
        if current_database_revision(settings.database_url) != "0005":
            raise ConfigurationError(
                "run requires database migration 0005; run stepik-agent db-upgrade first"
            )
        target = parse_stepik_url(url)
        token = TokenStore(settings.stepik_token_path).load()
        active_database = Database(settings.database_url)
        database = active_database

        async def execute_run() -> tuple[SyncResult, CourseRunResult]:
            shared_budget = SharedAIBudget(
                settings.max_run_cost_usd,
                max_calls=settings.max_llm_calls_per_run,
            )
            async with (
                StepikReadClient(
                    base_url=str(settings.stepik_base_url),
                    access_token=token.access_token,
                ) as read_client,
                create_anymodel_gateway(
                    settings,
                    request_observer=events.append,
                    shared_budget=shared_budget,
                ) as gateway,
                StepikWriteClient(
                    base_url=str(settings.stepik_base_url),
                    access_token=token.access_token,
                ) as write_client,
            ):
                if resume:
                    if target.course_id is None:
                        raise ConfigurationError("--resume requires a Stepik course URL")
                    sync_result = _local_resume_snapshot(active_database, target.course_id)
                else:
                    with active_database.session() as session:
                        sync_result = await CourseSyncService(
                            client=read_client,
                            writer=CatalogRepository(session),
                        ).sync(target)

                solver = DryRunService(
                    loader=StepikTaskLoader(read_client),
                    analyzer=TaskAnalyzer(),
                    gateway=gateway,
                    max_repair_attempts=settings.max_solution_repair_attempts,
                )
                submitter = SingleTaskSubmitService(
                    client=write_client,
                    store=AttemptStore(active_database),
                    verdict_timeout_seconds=settings.verdict_timeout_seconds,
                    max_attempts=settings.max_total_attempts_per_task,
                )

                def save_candidate(result: DryRunResult) -> None:
                    DryRunReportStore(output_dir).save(result, events)

                def record_interactive_error(
                    run_id: str,
                    step_id: int,
                    task_type: str,
                    reason: str,
                ) -> None:
                    with active_database.session() as session:
                        EventRepository(session).append(
                            run_id=run_id,
                            step_id=step_id,
                            event_type="interactive_task_blocked",
                            severity=EventSeverity.ERROR,
                            sanitized_payload={
                                "task_type": task_type,
                                "reason": reason[:1_000],
                            },
                        )

                code_executor = CodeTaskRunner(
                    solver=solver,
                    submitter=submitter,
                    max_attempts=settings.max_attempts_per_task,
                    max_total_attempts=settings.max_total_attempts_per_task,
                    model_id=settings.anymodel_model,
                    candidate_observer=save_candidate,
                    event_observer=record_interactive_error,
                )

                async def confirm_completion(course_id: int) -> bool:
                    with active_database.session() as session:
                        await CourseSyncService(
                            client=read_client,
                            writer=CatalogRepository(session),
                        ).sync(target)
                    return all(
                        task.is_passed
                        for task in CourseRunStore(active_database).list_tasks(course_id)
                    )

                run_result = await CourseRunner(
                    queue=CourseRunStore(active_database),
                    code_executor=code_executor,
                    cost_provider=lambda: shared_budget.known_cost_usd,
                    completion_checker=confirm_completion,
                    view_recorder=write_client,
                    blocked_reconciler=submitter,
                ).execute(sync_result.course_id)
                return sync_result, run_result

        with CourseRunLock(settings.database_url):
            sync_result, run_result = asyncio.run(execute_run())
    except (StepikAgentError, SQLAlchemyError, OSError, ValueError) as error:
        typer.echo(f"Course run failed: {error}", err=True)
        raise typer.Exit(code=2) from error
    finally:
        if database is not None:
            database.dispose()

    typer.echo(
        "Run completed "
        f"course={sync_result.course_id} "
        f"catalog_mode={'local_resume' if resume else 'full_sync'} "
        f"catalog_tasks={sync_result.tasks} "
        f"run_id={run_result.run_id} state={run_result.state.value} "
        f"passed={run_result.already_passed} completed={run_result.completed} "
        f"informational={run_result.informational} unsupported={run_result.unsupported} "
        f"failed={run_result.failed} blocked={run_result.blocked} "
        f"completion_confirmed={run_result.completion_confirmed}"
    )
    if sync_result.missing_assignment_ids:
        typer.echo(
            "WARNING: Stepik did not return optional assignment metadata for ids: "
            + ", ".join(str(value) for value in sync_result.missing_assignment_ids),
            err=True,
        )


def _local_resume_snapshot(database: Database, course_id: int) -> SyncResult:
    tasks = CourseRunStore(database).list_tasks(course_id)
    if not tasks:
        raise ConfigurationError(
            f"Course {course_id} has no synchronized local snapshot; run without --resume first"
        )
    return SyncResult(
        course_id=course_id,
        sections=0,
        units=0,
        lessons=0,
        tasks=len(tasks),
        passed_tasks=sum(
            task.is_passed
            or (
                task.status is TaskStatus.CORRECT
                and task.task_type not in {"text", "video", "animation"}
            )
            for task in tasks
        ),
    )


@app.command("dry-run")
def dry_run(
    url: str,
    allow_ai_cost: bool = typer.Option(
        False,
        "--allow-ai-cost",
        help="Explicitly allow bounded, potentially billable AnyModel calls.",
    ),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for the local JSON report."),
    ] = _DEFAULT_DRY_RUN_OUTPUT,
) -> None:
    """Generate one locally validated candidate without any Stepik POST request."""

    if not allow_ai_cost:
        typer.echo(
            "Dry run stopped before AI call: pass --allow-ai-cost to acknowledge possible cost.",
            err=True,
        )
        raise typer.Exit(code=2)

    events: list[AIRequestEvent] = []
    try:
        settings = Settings()
        target = parse_stepik_url(url)
        token_store = TokenStore(settings.stepik_token_path)
        access_token = token_store.load().access_token if token_store.exists else None

        async def run() -> DryRunResult:
            async with (
                StepikReadClient(
                    base_url=str(settings.stepik_base_url),
                    access_token=access_token,
                ) as client,
                create_anymodel_gateway(
                    settings,
                    request_observer=events.append,
                ) as gateway,
            ):
                service = DryRunService(
                    loader=StepikTaskLoader(client),
                    analyzer=TaskAnalyzer(),
                    gateway=gateway,
                    max_repair_attempts=settings.max_solution_repair_attempts,
                )
                return await service.execute(target)

        result = asyncio.run(run())
        report_path = DryRunReportStore(output_dir).save(result, events)
    except (StepikAgentError, OSError, ValueError) as error:
        typer.echo(f"Dry run failed: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(result.candidate.model_dump_json(indent=2))
    typer.echo(f"AI review: accepted; repair attempts: {result.repair_attempts}")
    for warning in result.warnings:
        typer.echo(f"WARNING: {warning}", err=True)
    if events and any(event.provider_cost_usd is None for event in events):
        typer.echo("WARNING: provider did not report exact cost for every AI request", err=True)
    typer.echo(f"Local report: {report_path}")
    typer.echo("Stepik submission: not performed")


@app.command("submit-one")
def submit_one(
    url: str,
    allow_ai_cost: bool = typer.Option(
        False,
        "--allow-ai-cost",
        help="Allow bounded, potentially billable AnyModel calls.",
    ),
    confirm_submit: bool = typer.Option(
        False,
        "--confirm-submit",
        help="Confirm one real Stepik attempt and submission.",
    ),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for the pre-submission JSON report."),
    ] = _DEFAULT_DRY_RUN_OUTPUT,
) -> None:
    """Generate, review, and submit one code answer with durable write state."""

    if not allow_ai_cost:
        typer.echo(
            "Submission stopped before AI call: pass --allow-ai-cost to acknowledge cost.",
            err=True,
        )
        raise typer.Exit(code=2)
    if not confirm_submit:
        typer.echo(
            "Submission stopped before network writes: pass --confirm-submit for one real answer.",
            err=True,
        )
        raise typer.Exit(code=2)

    database: Database | None = None
    report_path: Path | None = None
    events: list[AIRequestEvent] = []
    try:
        settings = Settings()
        if settings.dry_run or not settings.auto_submit:
            raise ConfigurationError("submit-one requires DRY_RUN=false and AUTO_SUBMIT=true")
        if current_database_revision(settings.database_url) != "0005":
            raise ConfigurationError(
                "submit-one requires database migration 0005; run stepik-agent db-upgrade first"
            )
        target = parse_stepik_url(url)
        active_database = Database(settings.database_url)
        database = active_database
        if target.lesson_id is None or target.step_position is None:
            raise ConfigurationError("submit-one requires one concrete lesson step URL")
        with active_database.session() as session:
            known_task = CatalogRepository(session).get_task_by_lesson_position(
                target.lesson_id,
                target.step_position,
            )
            if known_task is not None and known_task.is_passed:
                raise ConfigurationError(
                    f"Step {known_task.step_id} is already passed; duplicate submission blocked"
                )
        token = TokenStore(settings.stepik_token_path).load()

        def save_candidate(result: DryRunResult) -> None:
            nonlocal report_path
            report_path = DryRunReportStore(output_dir).save(result, events)

        async def execute_task() -> CodeTaskResult:
            async with (
                StepikReadClient(
                    base_url=str(settings.stepik_base_url),
                    access_token=token.access_token,
                ) as read_client,
                create_anymodel_gateway(
                    settings,
                    request_observer=events.append,
                ) as gateway,
                StepikWriteClient(
                    base_url=str(settings.stepik_base_url),
                    access_token=token.access_token,
                ) as write_client,
            ):
                solver = DryRunService(
                    loader=StepikTaskLoader(read_client),
                    analyzer=TaskAnalyzer(),
                    gateway=gateway,
                    max_repair_attempts=settings.max_solution_repair_attempts,
                )
                submitter = SingleTaskSubmitService(
                    client=write_client,
                    store=AttemptStore(active_database),
                    verdict_timeout_seconds=settings.verdict_timeout_seconds,
                    max_attempts=settings.max_total_attempts_per_task,
                )
                return await CodeTaskRunner(
                    solver=solver,
                    submitter=submitter,
                    max_attempts=settings.max_attempts_per_task,
                    max_total_attempts=settings.max_total_attempts_per_task,
                    model_id=settings.anymodel_model,
                    candidate_observer=save_candidate,
                ).execute(target)

        task_result = asyncio.run(execute_task())
    except (StepikAgentError, SQLAlchemyError, OSError, ValueError) as error:
        if report_path is not None:
            typer.echo(f"Pre-submission report: {report_path}", err=True)
        typer.echo(f"Single submission failed: {error}", err=True)
        raise typer.Exit(code=2) from error
    finally:
        if database is not None:
            database.dispose()

    assert report_path is not None
    attempt = task_result.attempt
    typer.echo(f"Pre-submission report: {report_path}")
    typer.echo(f"Local attempt: {attempt.local_attempt_id}")
    typer.echo(f"Stepik attempt: {attempt.stepik_attempt_id}")
    typer.echo(f"Stepik submission: {attempt.stepik_submission_id}")
    typer.echo(f"Stepik verdict: {attempt.verdict}")
    typer.echo(f"Submitted attempts this run: {task_result.submitted_attempts}")


@app.command("resume-submit")
def resume_submit(
    local_attempt_id: int,
    confirm_submit: bool = typer.Option(
        False,
        "--confirm-submit",
        help="Allow safe continuation; a persisted pre-submission state may perform one POST.",
    ),
) -> None:
    """Resume one persisted submission without replaying an ambiguous POST."""

    if not confirm_submit:
        typer.echo(
            "Resume stopped: pass --confirm-submit because continuation may perform one POST.",
            err=True,
        )
        raise typer.Exit(code=2)

    database: Database | None = None
    try:
        settings = Settings()
        if settings.dry_run or not settings.auto_submit:
            raise ConfigurationError("resume-submit requires DRY_RUN=false and AUTO_SUBMIT=true")
        if current_database_revision(settings.database_url) != "0005":
            raise ConfigurationError(
                "resume-submit requires database migration 0005; run stepik-agent db-upgrade first"
            )
        token = TokenStore(settings.stepik_token_path).load()
        active_database = Database(settings.database_url)
        database = active_database

        async def resume() -> AttemptRecord:
            async with StepikWriteClient(
                base_url=str(settings.stepik_base_url),
                access_token=token.access_token,
            ) as write_client:
                return await SingleTaskSubmitService(
                    client=write_client,
                    store=AttemptStore(active_database),
                    verdict_timeout_seconds=settings.verdict_timeout_seconds,
                    max_attempts=settings.max_total_attempts_per_task,
                ).resume(local_attempt_id)

        attempt = asyncio.run(resume())
    except (StepikAgentError, SQLAlchemyError, OSError, ValueError) as error:
        typer.echo(f"Submission resume failed: {error}", err=True)
        raise typer.Exit(code=2) from error
    finally:
        if database is not None:
            database.dispose()

    typer.echo(f"Local attempt: {attempt.local_attempt_id}")
    typer.echo(f"State: {attempt.state.value}")
    typer.echo(f"Stepik attempt: {attempt.stepik_attempt_id}")
    typer.echo(f"Stepik submission: {attempt.stepik_submission_id}")
    typer.echo(f"Stepik verdict: {attempt.verdict}")


@app.command("ai-smoke")
def ai_smoke(
    allow_ai_cost: bool = typer.Option(
        False,
        "--allow-ai-cost",
        help="Explicitly allow one bounded, potentially billable AnyModel check.",
    ),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for the local JSON report."),
    ] = _DEFAULT_AI_SMOKE_OUTPUT,
) -> None:
    """Validate AnyModel connectivity and JSON output without contacting Stepik."""

    if not allow_ai_cost:
        typer.echo(
            "AI smoke stopped before network call: pass --allow-ai-cost to acknowledge cost.",
            err=True,
        )
        raise typer.Exit(code=2)

    events: list[AIRequestEvent] = []
    try:
        settings = Settings()

        async def run() -> DryRunResult:
            async with create_anymodel_gateway(
                settings,
                request_observer=events.append,
            ) as gateway:
                return await run_ai_smoke(gateway)

        result = asyncio.run(run())
        report_path = DryRunReportStore(output_dir).save(result, events, mode="ai_smoke")
    except (StepikAgentError, OSError, ValueError) as error:
        typer.echo(f"AI smoke failed: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(result.candidate.model_dump_json(indent=2))
    if events and any(event.provider_cost_usd is None for event in events):
        typer.echo("WARNING: provider did not report exact cost", err=True)
    typer.echo(f"Local report: {report_path}")
    typer.echo("Stepik network: not used")
