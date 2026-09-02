"""Asynchronous database engine, session management, and dependencies."""
import logging
from typing import AsyncGenerator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from backend.app.core.config import settings
from backend.app.models.base import Base

logger = logging.getLogger("cv_rag_pipeline.db")

# Create asynchronous SQLAlchemy engine
def get_engine(db_url: str = settings.SUPABASE_DB_URL) -> AsyncEngine:
    connect_args = {}
    if "sqlite" in db_url:
        connect_args["check_same_thread"] = False
        return create_async_engine(
            db_url,
            echo=False,
            connect_args=connect_args,
        )
    return create_async_engine(
        db_url,
        echo=False,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
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
    async with target_engine.begin() as conn:
        # Enable pgvector if on PostgreSQL
        if "postgresql" in str(target_engine.url):
            try:
                logger.info("Enabling pgvector extension on PostgreSQL...")
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                logger.info("pgvector extension verified.")
            except Exception as e:
                logger.warning(f"Could not create vector extension (may lack superuser permissions or already enabled): {e}")

        # Create all tables defined in models
        logger.info("Creating database tables if not present...")
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database schema initialized successfully.")
