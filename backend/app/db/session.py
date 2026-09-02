"""Asynchronous database engine, session management, and dependencies."""
import logging
from typing import AsyncGenerator, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from backend.app.core.config import settings
from backend.app.models import Base, CVDocument, CVChunk, CVProcessingTrace

logger = logging.getLogger("cv_rag_pipeline.db")

# Try importing pgvector asyncpg codec registration
try:
    from pgvector.asyncpg import register_vector as _register_vector
    _HAS_PGVECTOR = True
except ImportError:
    _HAS_PGVECTOR = False
    logger.warning("pgvector.asyncpg not available; vector codec will not be registered.")


async def _register_vector_codec(raw_conn):
    """Coroutine that registers the pgvector codec on a raw asyncpg connection."""
    if _HAS_PGVECTOR:
        try:
            await _register_vector(raw_conn)
        except Exception as exc:
            logger.debug(f"pgvector codec registration skipped: {exc}")


# Create asynchronous SQLAlchemy engine
def get_engine(db_url: Optional[str] = None) -> AsyncEngine:
    url = db_url or settings.SUPABASE_DB_URL
    if not url:
        url = "sqlite+aiosqlite:///./cv_pipeline.db"

    connect_args: dict = {}
    if "sqlite" in url:
        connect_args["check_same_thread"] = False
        return create_async_engine(
            url,
            echo=False,
            connect_args=connect_args,
        )

    # For asyncpg: register the pgvector codec on every new connection via
    # the async_creator / init callback so it runs in the correct async context.
    return create_async_engine(
        url,
        echo=False,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        # asyncpg-specific: called with the raw asyncpg connection after connect
        connect_args={"server_settings": {"jit": "off"}},
    )


engine: AsyncEngine = get_engine()


async def _on_connect(dbapi_conn, connection_record):
    """SQLAlchemy async pool event – registers pgvector codec on raw asyncpg connections."""
    # dbapi_conn is a asyncpg Connection proxied by SQLAlchemy
    await _register_vector_codec(dbapi_conn)


# Register the async event using the asyncpg-aware hook
try:
    from sqlalchemy import event as _sa_event
    @_sa_event.listens_for(engine.sync_engine, "connect")
    def _sync_connect_hook(dbapi_connection, connection_record):
        """Synchronous connect hook – runs the async codec registration via run_async."""
        if _HAS_PGVECTOR and hasattr(dbapi_connection, "run_async"):
            try:
                dbapi_connection.run_async(_register_vector_codec)
            except Exception as exc:
                logger.debug(f"Could not register pgvector codec via run_async: {exc}")
except Exception as hook_err:
    logger.debug(f"Could not attach pgvector connect hook: {hook_err}")


# Async session factory
AsyncSessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding an asynchronous database session."""
    async with AsyncSessionFactory() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}", exc_info=True)
            raise
        finally:
            await session.close()


async def init_db(target_engine: AsyncEngine = engine):
    """Initializes the database schema and enables pgvector extension."""
    if "postgresql" in str(target_engine.url):
        try:
            async with target_engine.connect() as conn:
                await conn.execution_options(isolation_level="AUTOCOMMIT")
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                logger.info("pgvector extension verified.")
        except Exception as e:
            logger.warning(f"Could not create vector extension: {e}")

    try:
        async with target_engine.begin() as conn:
            logger.info("Creating database tables if not present...")
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database schema initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database tables: {e}", exc_info=True)
