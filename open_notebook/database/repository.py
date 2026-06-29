import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypeVar, Union

from loguru import logger
from surrealdb import AsyncSurreal, RecordID  # type: ignore

T = TypeVar("T", Dict[str, Any], List[Dict[str, Any]])

# ─── Connection Pool ─────────────────────────────────────────────────────────
# Reuses a single connection across the API lifespan to avoid the overhead of
# opening a new WebSocket connection per query. Reconnects automatically on
# disconnect.

_pool_connection: Optional[AsyncSurreal] = None
_pool_config: Optional[dict] = None


def _get_pool_config() -> dict:
    global _pool_config
    if _pool_config is None:
        _pool_config = {
            "url": get_database_url(),
            "user": os.environ.get("SURREAL_USER", "root"),
            "password": get_database_password(),
            "namespace": get_database_namespace(),
            "database": get_database_name(),
        }
    return _pool_config


async def get_connection() -> AsyncSurreal:
    """Get or create a shared database connection with auto-reconnect."""
    global _pool_connection
    cfg = _get_pool_config()

    if _pool_connection is None:
        _pool_connection = AsyncSurreal(cfg["url"])
        await _pool_connection.signin({"username": cfg["user"], "password": cfg["password"]})
        await _pool_connection.use(cfg["namespace"], cfg["database"])
        logger.debug(f"DB connected: {cfg['url']} / {cfg['namespace']}.{cfg['database']}")
    else:
        # Health check — reconnect if closed
        try:
            await _pool_connection.health()
        except Exception:
            logger.warning("DB connection lost — reconnecting...")
            _pool_connection = AsyncSurreal(cfg["url"])
            await _pool_connection.signin({"username": cfg["user"], "password": cfg["password"]})
            await _pool_connection.use(cfg["namespace"], cfg["database"])
            logger.info("DB reconnected")

    return _pool_connection


async def close_connection() -> None:
    """Close the shared database connection. Call on app shutdown."""
    global _pool_connection
    if _pool_connection is not None:
        try:
            await _pool_connection.close()
        except Exception:
            pass
        _pool_connection = None
        logger.debug("DB connection closed")


@asynccontextmanager
async def db_connection():
    """Legacy context manager — uses the shared pool internally."""
    db = await get_connection()
    try:
        yield db
    except Exception:
        # On error, force reconnect for next caller
        await close_connection()
        raise


def _get_env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


def get_database_url() -> str:
    """Get database URL with backward compatibility"""
    surreal_url = os.getenv("SURREAL_URL")
    if surreal_url:
        return surreal_url

    # Fallback to old format - WebSocket URL format
    address = os.getenv("SURREAL_ADDRESS", "localhost")
    port = os.getenv("SURREAL_PORT", "8000")
    return f"ws://{address}/rpc:{port}"


def get_database_password() -> str:
    """Get password with backward compatibility"""
    return os.getenv("SURREAL_PASSWORD") or os.getenv("SURREAL_PASS") or "root"


def get_database_namespace() -> str:
    """Get configured SurrealDB namespace."""
    return _get_env_or_default("SURREAL_NAMESPACE", "open_notebook")


def get_database_name() -> str:
    """Get configured SurrealDB database name."""
    return _get_env_or_default("SURREAL_DATABASE", "open_notebook")


def parse_record_ids(obj: Any) -> Any:
    """Recursively parse and convert RecordIDs into strings."""
    if isinstance(obj, dict):
        return {k: parse_record_ids(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [parse_record_ids(item) for item in obj]
    elif isinstance(obj, RecordID):
        return str(obj)
    return obj


def ensure_record_id(value: Union[str, RecordID]) -> RecordID:
    """Ensure a value is a RecordID."""
    if isinstance(value, RecordID):
        return value
    return RecordID.parse(value)


async def repo_query(
    query_str: str, vars: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    db = await get_connection()
    try:
        result = await db.query(query_str, vars)
        # SurrealDB returns [[...]] for query()
        if isinstance(result, list) and len(result) > 0:
            return result[0] if isinstance(result[0], list) else result
        return result or []
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise


async def repo_create(
    table: str, data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    db = await get_connection()
    data["created"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data["updated"] = data["created"]
    if "id" in data:
        data.pop("id")
    try:
        result = await db.create(table, data)
        return result[0] if isinstance(result, list) else result
    except Exception as e:
        logger.error(f"Create failed for {table}: {e}")
        raise


async def repo_insert(
    table: str, data_list: List[Dict[str, Any]], ignore_duplicates: bool = False
) -> Optional[List[Dict[str, Any]]]:
    if not data_list:
        return []
    db = await get_connection()
    try:
        result = await db.insert(table, data_list)
        return result if isinstance(result, list) else [result]
    except RuntimeError as e:
        if ignore_duplicates and "already contains" in str(e):
            return []
        raise
    except Exception as e:
        logger.error(f"Insert failed for {table}: {e}")
        if ignore_duplicates:
            return []
        raise


async def repo_upsert(
    table: str, id: Union[str, RecordID], data: Dict[str, Any], add_timestamp: bool = True
) -> Optional[Dict[str, Any]]:
    db = await get_connection()
    if add_timestamp:
        data["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        result = await db.upsert(
            {
                "table": table,
                "id": id if isinstance(id, RecordID) else RecordID.parse(id),
                "data": data,
            }
        )
        return result[0] if isinstance(result, list) and len(result) == 1 else result
    except Exception as e:
        logger.error(f"Upsert failed for {table}/{id}: {e}")
        raise


async def repo_update(
    table_or_id: str, id_or_data: Any = None, data: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    db = await get_connection()
    if data is None:
        record_id = ensure_record_id(table_or_id)
        update_data = id_or_data
    else:
        record_id = ensure_record_id(f"{table_or_id}:{id_or_data}")
        update_data = data

    update_data["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Preserve created timestamp from existing record
    if "created" in update_data and isinstance(update_data["created"], str):
        try:
            update_data["created"] = datetime.fromisoformat(update_data["created"])
        except ValueError:
            pass

    try:
        result = await db.merge(record_id, update_data)
        return result
    except Exception as e:
        logger.error(f"Update failed for {record_id}: {e}")
        raise


async def repo_delete(record_id: Union[str, RecordID]) -> None:
    db = await get_connection()
    rid = ensure_record_id(record_id)
    try:
        await db.delete(rid)
    except Exception as e:
        logger.error(f"Delete failed for {record_id}: {e}")
        raise


async def repo_relate(
    source: str, relationship: str, target: str, data: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    db = await get_connection()
    try:
        result = await db.query(
            "RELATE $source -> $relationship -> $target CONTENT $data",
            {
                "source": ensure_record_id(source),
                "relationship": relationship,
                "target": ensure_record_id(target),
                "data": data or {},
            },
        )
        return result[0][0] if (result and isinstance(result[0], list) and result[0]) else None
    except Exception as e:
        logger.error(f"Relate failed: {e}")
        raise
