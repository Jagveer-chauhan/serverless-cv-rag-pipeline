"""Asynchronous database engine, session management, and dependencies."""
import logging
from typing import AsyncGenerator, Optional
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from backend.app.core.config import settings
from backend.app.models import Base, CVDocument, CVChunk, CVProcessingTrace

logger = logging.getLogger("cv_rag_pipeline.db")

# Create asynchronous SQLAlchemy engine
def get_engine(db_url: Optional[str] = None) -> AsyncEngine:
    url = db_url or settings.SUPABASE_DB_URL
    if not url:
        url = "sqlite+aiosqlite:///./cv_pipeline.db"

    connect_args = {}
    if "sqlite" in url:
        connect_args["check_same_thread"] = False
        return create_async_engine(
            url,
            echo=False,
            connect_args=connect_args,
        )
    return create_async_engine(
        url,
        echo=False,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )


engine: AsyncEngine = get_engine()

# Register pgvector codec for asyncpg connections
try:
    from pgvector.asyncpg import register_vector
except ImportError:
    register_vector = None


@event.listens_for(engine.sync_engine, "connect")
def register_custom_types(dbapi_connection, connection_record):
    """Registers asyncpg vector(384) decoder on every new database connection."""
    if register_vector and hasattr(dbapi_connection, "run_async"):
        dbapi_connection.run_async(lambda conn: register_vector(conn))


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
