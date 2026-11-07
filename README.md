# Stepik Autonomous Agent

Локальный CLI-агент для работы с поддерживаемыми учебными заданиями Stepik. Агент получает
структуру курса и прогресс через Stepik API, при необходимости запрашивает решение у
OpenAI-compatible AI Gateway (AnyModel), валидирует ответ и сохраняет состояние в SQLite.

Проект рассчитан на локальный запуск под контролем владельца аккаунта. Он не обходит CAPTCHA,
ограничения доступа или лимиты платформы и не должен использоваться для курсов или аккаунтов,
для которых автоматизация запрещена правилами Stepik.

## Состояние проекта

Текущий инкремент реализует безопасный локальный контур и CLI-команды для синхронизации,
просмотра состояния, dry-run и явного запуска отправок. Подробная история решений и результаты
последних проверок находятся в [PROJECT_STATUS.md](PROJECT_STATUS.md).

Автоматическая отправка по умолчанию выключена. Любой вызов, который может создать attempt,
submission или изменить прогресс Stepik, требует отдельной настройки и явного флага CLI.

## Возможности

- OAuth 2 Authorization Code для Stepik с loopback callback и локальным хранением токена с
  правами `0600`.
- Read-only синхронизация course → sections → units → lessons → steps → progress с пагинацией.
- SQLite и Alembic для сохранения курса, попыток, запусков, вердиктов, событий и расходов AI.
- Dry-run одного шага без `POST` в Stepik; отдельная проверка AnyModel без обращения к Stepik.
- Возобновление локального запуска курса после прерывания и защита от параллельных запусков.
- Ограничения числа AI-вызовов, попыток и бюджета на задачу и на весь запуск.
- Роли Solver, Reviewer и Repairer с Pydantic-валидацией структурированного ответа.
- Долговечное состояние отправки и сверка неоднозначного `POST`, чтобы не повторять его вслепую.

Поддерживаемые write-типы в текущей версии: `code`, `number`, `string`, `math`, `choice`,
`matching`, `fill-blanks`, `table`, `dataset`. Информационные `text`, `video` и `animation`
отмечаются просмотренными через Stepik API без AI. Неподдерживаемые, недоступные или требующие
ручной проверки задания остаются явно помеченными и не выдаются за выполненные.

## Требования

- Python 3.11 или новее;
- OAuth-приложение Stepik с Authorization Code grant;
- API-ключ и модель провайдера AnyModel только для команд, использующих AI;
- доступ к сети для `auth`, `sync`, `dry-run` и живых запусков.

Для разработки и тестов достаточно Python-зависимостей из optional-группы `dev`. Сгенерированный
или пользовательский код агент не выполняет автоматически.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
touch .env
```

Создайте `.env` вручную по инструкции ниже. Этот файл содержит секреты, уже исключён из Git и
не должен публиковаться.

Затем примените миграции и проверьте конфигурацию:

```bash
stepik-agent db-upgrade
stepik-agent doctor
stepik-agent version
```

Если editable-установка на macOS не видит пакет, используйте обычную установку wheel:

```bash
python -m pip install --force-reinstall '.[dev]'
python -c 'import stepik_agent'
```

## Конфигурация `.env`

Откройте файл в редакторе:

```bash
${EDITOR:-nano} .env
```

Минимальная конфигурация для безопасного read-only запуска:

```dotenv
# Stepik OAuth
STEPIK_BASE_URL=https://stepik.org
STEPIK_CLIENT_ID=вставьте_client_id_своего_OAuth_приложения
STEPIK_CLIENT_SECRET=вставьте_client_secret_своего_OAuth_приложения
STEPIK_REDIRECT_URI=http://127.0.0.1:8080/callback
STEPIK_TOKEN_PATH=data/stepik_token.json

# Локальное состояние
DATABASE_URL=sqlite:///./data/agent.db

# Безопасный режим: отправки выключены
DRY_RUN=true
AUTO_SUBMIT=false
```

Для команд `ai-smoke` и `dry-run` добавьте настройки AnyModel:

```dotenv
ANYMODEL_BASE_URL=https://anymodel.org/v1
ANYMODEL_API_KEY=ваш_private_api_key
ANYMODEL_MODEL=идентификатор_модели
ANYMODEL_REVIEW_MODEL=
ANYMODEL_FALLBACK_MODEL=
ANYMODEL_MAX_OUTPUT_TOKENS=4096
ANYMODEL_VALIDATION_RETRIES=1
ANYMODEL_JSON_MODE=true
```

Ограничения расходов и повторов рекомендуется задать явно:

```dotenv
MAX_ATTEMPTS_PER_TASK=5
MAX_TOTAL_ATTEMPTS_PER_TASK=25
MAX_LLM_CALLS_PER_TASK=8
MAX_LLM_CALLS_PER_RUN=500
MAX_SOLUTION_REPAIR_ATTEMPTS=2
MAX_TASK_COST_USD=1.00
MAX_RUN_COST_USD=5.00
AI_REQUEST_TIMEOUT_SECONDS=60
VERDICT_TIMEOUT_SECONDS=300
```

Пустые значения для `STEPIK_CLIENT_SECRET` и `ANYMODEL_API_KEY` считаются ненастроенными.
Никогда не вставляйте эти значения в README, исходники, тесты, issue или логи.

## Настройка Stepik

1. Создайте OAuth-приложение в [настройках Stepik](https://stepik.org/oauth2/applications/).
2. Укажите в `.env` `STEPIK_CLIENT_ID`, `STEPIK_CLIENT_SECRET` и loopback-адрес
   `STEPIK_REDIRECT_URI`.
3. Выполните авторизацию:

```bash
stepik-agent auth
# или выведите URL без автоматического открытия браузера
stepik-agent auth --no-browser
```

Токен сохраняется в `data/stepik_token.json`. Не публикуйте его, client secret, authorization
code или содержимое локальных отчётов.

Проверить URL без сети можно так:

```bash
stepik-agent inspect-url \
  'https://stepik.org/lesson/41871/step/1?unit=20186'
```

Read-only синхронизация курса:

```bash
stepik-agent sync 'https://stepik.org/course/COURSE_ID'
```

Команда `sync` выполняет GET-запросы и локальные записи в SQLite; попытки и отправки в Stepik
она не создаёт.

## Настройка AnyModel

Для AI-команд заполните в `.env` минимум:

```dotenv
ANYMODEL_BASE_URL=https://anymodel.org/v1
ANYMODEL_API_KEY=your_private_api_key
ANYMODEL_MODEL=your_model_id
```

Ограничения расходов и вызовов задаются переменными `MAX_TASK_COST_USD`,
`MAX_RUN_COST_USD`, `MAX_LLM_CALLS_PER_TASK` и `MAX_LLM_CALLS_PER_RUN`. Провайдер может не
вернуть точную стоимость; в этом случае отчёт помечает стоимость как неизвестную, а лимит числа
вызовов всё равно действует.

## Основные команды

| Команда | Назначение | Изменяет прогресс Stepik |
| --- | --- | --- |
| `stepik-agent doctor` | Проверить наличие настроек без раскрытия секретов | Нет |
| `stepik-agent status` | Показать последний локальный запуск | Нет |
| `stepik-agent db-upgrade` | Применить миграции SQLite | Только локальная БД |
| `stepik-agent auth` | Получить OAuth-токен Stepik | Нет |
| `stepik-agent sync URL` | Синхронизировать курс и прогресс | Нет |
| `stepik-agent ai-smoke --allow-ai-cost` | Проверить AnyModel на синтетической задаче | Нет |
| `stepik-agent dry-run URL --allow-ai-cost` | Получить и проверить один кандидат без `POST` | Нет |
| `stepik-agent submit-one URL --allow-ai-cost --confirm-submit` | Отправить один ответ | Да |
| `stepik-agent run URL --allow-ai-cost --confirm-submit` | Выполнить поддержанные задачи курса | Да |
| `stepik-agent run URL --resume ...` | Продолжить последний локальный snapshot курса | Возможно |
| `stepik-agent resume-submit ID --confirm-submit` | Безопасно продолжить сохранённую отправку | Возможно |

До первого реального запуска рекомендуется пройти read-only этап:

```bash
stepik-agent sync 'https://stepik.org/course/COURSE_ID'
stepik-agent ai-smoke --allow-ai-cost
stepik-agent dry-run \
  'https://stepik.org/lesson/LESSON_ID/step/POSITION?unit=UNIT_ID' \
  --allow-ai-cost
```

Для живой отправки должны быть явно выставлены `DRY_RUN=false` и `AUTO_SUBMIT=true`. Команды
`submit-one` и `run` дополнительно требуют `--confirm-submit` и `--allow-ai-cost`.

Пример запуска курса:

```bash
stepik-agent run 'https://stepik.org/course/COURSE_ID' \
  --allow-ai-cost \
  --confirm-submit
```

Если процесс прервался, сначала проверьте `stepik-agent status`, затем используйте `--resume`.
Не запускайте несколько процессов на одной локальной БД одновременно.

## Хранение данных и безопасность

Локальные токены, SQLite, отчёты dry-run/AI-smoke и lock-файлы находятся в `data/` и исключены
из Git. Секреты не должны попадать в исходники, тесты, issue, логи или скриншоты.

Перед отправкой ответа агент сохраняет локальное состояние и ограничивает повторы. Timeout или
неоднозначный ответ Stepik не считается успешной отправкой: состояние остаётся доступным для
последующей read-only сверки и `resume-submit`.

Перед использованием на конкретном курсе проверьте правила курса и платформы. Пользователь
самостоятельно подтверждает допустимость автоматизации, расходы AnyModel и изменение прогресса.

## Структура проекта

```text
src/stepik_agent/
  ai/            AnyModel gateway, бюджеты и модели ответов
  application/   dry-run, одиночная отправка и запуск курса
  core/          состояния и доменные ошибки
  stepik/        OAuth, read/write API, синхронизация и загрузчик задач
  storage/       SQLite, репозитории, попытки, миграции и lock
  tasks/         адаптеры типов заданий и локальная валидация
tests/            unit и integration тесты с mock/fake transport
migrations/       Alembic-миграции
```

`PROJECT_STATUS.md` — внутренний журнал разработки; для запуска приложения он не нужен и может
не включаться в публичный репозиторий.

## Проверки разработчика

В окружении с установленными dev-зависимостями выполните:

```bash
pytest
ruff check .
ruff format --check .
mypy src/stepik_agent
python -m compileall -q src migrations
python -m pip check
```

Тесты не должны обращаться к реальному Stepik или списывать средства AnyModel. Живые `auth`,
`doctor`, `sync`, `dry-run`, `submit-one` и `run` запускает только пользователь в своём
терминале с локальной конфигурацией.

## Лицензия

В `pyproject.toml` сейчас указана лицензия `Proprietary`. Перед публичной публикацией выберите
лицензию проекта и замените это значение, если код планируется распространять как open source.

## Публикация в Git

Перед первым push:

1. Убедитесь, что `.env`, токены, SQLite, отчёты и логи не индексируются.
2. Запустите все проверки из раздела выше.
3. Просмотрите будущий diff и список файлов вручную; не добавляйте `data/` целиком.
4. Решите, нужны ли в публичном репозитории `AGENTS.md` и `PROJECT_STATUS.md`.
5. Выберите лицензию и добавьте её файл, если репозиторий будет публичным.
6. Инициализируйте Git, создайте первый коммит и только затем настройте remote.

Команды Git выполняйте самостоятельно после проверки содержимого:

```bash
git init
git add .gitignore README.md pyproject.toml alembic.ini migrations src tests data/.gitkeep
git diff --cached --check
git diff --cached --stat
git commit -m "Initial project snapshot"
git branch -M main
git remote add origin <URL_ВАШЕГО_РЕПОЗИТОРИЯ>
git push -u origin main
```

Команды `git init`, `commit` и `push` в рамках подготовки этого изменения не выполнялись.
