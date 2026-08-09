import os
import re
import sys
import urllib.parse
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# Ensure src is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import settings
from src.db_models import Base


# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata for autogenerate support
target_metadata = Base.metadata


def sanitize_db_url(url: str) -> str:
    """Sanitize database URL by stripping pgbouncer flags and URL-encoding password special chars."""
    if not url or "://" not in url:
        return url
    if "pgbouncer=true" in url:
        url = url.replace("?pgbouncer=true&", "?").replace("&pgbouncer=true", "").replace("?pgbouncer=true", "")
    
    # URL-encode un-encoded special characters in password
    pattern = r"^(?P<scheme>[^:]+://)(?P<user>[^:]+):(?P<password>.+)@(?P<host>[^@:]+:\d+/.+)$"
    m = re.match(pattern, url)
    if m:
        scheme = m.group("scheme")
        user = m.group("user")
        password = m.group("password")
        host = m.group("host")
        if "%" not in password:
            password = urllib.parse.quote(password, safe="")
        return f"{scheme}{user}:{password}@{host}"
    return url


# Dynamically inject DATABASE_URL from config.env / settings
db_url = sanitize_db_url(settings.database_url or os.getenv("DATABASE_URL", ""))

if db_url:
    # Escape % as %% for configparser interpolation
    config.set_main_option("sqlalchemy.url", db_url.replace("%", "%%"))


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    context.configure(
        url=db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    configuration = config.get_section(config.config_ini_section, {})
    if db_url:
        configuration["sqlalchemy.url"] = db_url

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
