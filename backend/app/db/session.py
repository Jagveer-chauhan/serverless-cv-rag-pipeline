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


def get_engine(db_url: Optional[str] = None) -> AsyncEngine:
    """Creates an async SQLAlchemy engine for PostgreSQL (asyncpg) or SQLite (aiosqlite)."""
    url = db_url or settings.SUPABASE_DB_URL
    if not url:
        url = "sqlite+aiosqlite:///./cv_pipeline.db"

    if "sqlite" in url:
        return create_async_engine(
            url,
            echo=False,
            connect_args={"check_same_thread": False},
        )

    # PostgreSQL via asyncpg.
    # NOTE: We intentionally do NOT register the pgvector asyncpg codec here.
    # The pipeline uses raw SQL with CAST(:embedding AS vector) instead of the
    # ORM Vector column, so no codec is needed for INSERT.  The codec hook via
    # run_async is unreliable on free-tier Render/Supabase pooled connections
    # and was the root cause of the recurring DataError.
    return create_async_engine(
        url,
        echo=False,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        connect_args={"server_settings": {"jit": "off"}},
    )


engine: AsyncEngine = get_engine()

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


async def reset_db(target_engine: AsyncEngine = engine):
    """Drops and recreates all tables. USE WITH CAUTION — deletes all data."""
    async with target_engine.begin() as conn:
        logger.warning("Dropping all database tables for reset...")
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database reset complete — all tables recreated.")
