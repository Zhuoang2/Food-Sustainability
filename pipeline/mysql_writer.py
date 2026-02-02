"""
MySQL write: extraction_runs, restaurants, menu_items, ingredients,
menu_item_ingredient_observations, and sync menu_item_ingredients (current snapshot).
All in one transaction; rollback on error.
"""
import logging
from typing import List, Dict, Any, Optional

from .ingredients import canonicalize_ingredient, display_name_for_ingredient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection (caller can pass config or use env)
# ---------------------------------------------------------------------------


def get_mysql_config() -> Dict[str, Any]:
    import os
    import sys
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    password = os.environ.get("MYSQL_PASSWORD")
    if password is None:
        try:
            import config as _cfg
            password = getattr(_cfg, "MYSQL_PASSWORD", "") or ""
        except ImportError:
            password = ""
    return {
        "host": os.environ.get("MYSQL_HOST", "localhost"),
        "port": int(os.environ.get("MYSQL_PORT", "3306")),
        "user": os.environ.get("MYSQL_USER", "menu_user"),
        "password": password,
        "database": os.environ.get("MYSQL_DATABASE", "menu_llm"),
        "autocommit": False,
    }


def _get_pipeline_config():
    """从 config.py 读取 prompt_version、pipeline_version（未设置则 None）。"""
    import os
    import sys
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    try:
        import config as _cfg
        return getattr(_cfg, "PROMPT_VERSION", None), getattr(_cfg, "PIPELINE_VERSION", None)
    except ImportError:
        return None, None


def write_run(
    conn,
    results: List[Dict[str, Any]],
    *,
    model_name: Optional[str] = "gpt-4o",
    prompt_version: Optional[str] = None,
    pipeline_version: Optional[str] = None,
) -> int:
    """
    Write one extraction run: insert extraction_runs, upsert restaurants/menu_items,
    insert-or-get ingredients, insert observations, sync menu_item_ingredients.
    Returns run_id. Caller must commit or rollback.
    """
    if prompt_version is None or pipeline_version is None:
        _pv, _pipv = _get_pipeline_config()
        if prompt_version is None:
            prompt_version = _pv
        if pipeline_version is None:
            pipeline_version = _pipv
    cursor = conn.cursor()
    try:
        # 1) Insert extraction_runs（无 reasoning 列）
        cursor.execute(
            """
            INSERT INTO extraction_runs (model_name, prompt_version, pipeline_version)
            VALUES (%s, %s, %s)
            """,
            (model_name, prompt_version, pipeline_version),
        )
        run_id = cursor.lastrowid
        if not run_id:
            cursor.execute("SELECT LAST_INSERT_ID()")
            run_id = cursor.fetchone()[0]
        logger.info("Inserted extraction_runs run_id=%s", run_id)

        for item in results:
            item_id = int(item["item_id"])
            restaurant_id = int(item["restaurant_id"])
            item_name = item.get("item_name") or ""
            reasoning = item.get("reasoning")

            # 2) restaurants: insert if not exists (name NULL)
            cursor.execute(
                """
                INSERT INTO restaurants (restaurant_id, name)
                VALUES (%s, NULL)
                ON DUPLICATE KEY UPDATE updated_at = CURRENT_TIMESTAMP
                """,
                (restaurant_id,),
            )

            # 3) menu_items: upsert
            cursor.execute(
                """
                INSERT INTO menu_items (item_id, restaurant_id, item_name, reasoning)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                  restaurant_id = VALUES(restaurant_id),
                  item_name = VALUES(item_name),
                  reasoning = VALUES(reasoning),
                  updated_at = CURRENT_TIMESTAMP
                """,
                (item_id, restaurant_id, item_name, reasoning),
            )

            # 4) ingredients: insert-ignore by canonical_name, then get ingredient_id
            # 5) menu_item_ingredient_observations: insert (run_id, item_id, ingredient_id, confidence)
            # 6) sync menu_item_ingredients: delete old for this item_id, insert new
            ingredient_rows = item.get("ingredients") or []
            seen_ingredient_ids = []

            for ing in ingredient_rows:
                name = (ing.get("name") or "").strip()
                if not name:
                    continue
                conf = float(ing.get("confidence", 0))
                conf = max(0.0, min(1.0, conf))

                canonical = canonicalize_ingredient(name)
                display = display_name_for_ingredient(name)
                if not canonical:
                    continue

                cursor.execute(
                    """
                    INSERT INTO ingredients (canonical_name, display_name)
                    VALUES (%s, %s)
                    ON DUPLICATE KEY UPDATE display_name = VALUES(display_name)
                    """,
                    (canonical, display),
                )
                cursor.execute(
                    "SELECT ingredient_id FROM ingredients WHERE canonical_name = %s",
                    (canonical,),
                )
                row = cursor.fetchone()
                if not row:
                    raise RuntimeError(f"Failed to resolve ingredient_id for canonical_name={canonical!r}")
                ingredient_id = row[0]
                seen_ingredient_ids.append((ingredient_id, conf))

                cursor.execute(
                    """
                    INSERT INTO menu_item_ingredient_observations (run_id, item_id, ingredient_id, confidence)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE confidence = VALUES(confidence), created_at = CURRENT_TIMESTAMP
                    """,
                    (run_id, item_id, ingredient_id, round(conf, 3)),
                )

            # 6) Sync current snapshot: remove old links for this item, insert new
            cursor.execute("DELETE FROM menu_item_ingredients WHERE item_id = %s", (item_id,))
            for ing_id, conf in seen_ingredient_ids:
                cursor.execute(
                    """
                    INSERT INTO menu_item_ingredients (item_id, ingredient_id, confidence)
                    VALUES (%s, %s, %s)
                    """,
                    (item_id, ing_id, round(conf, 3)),
                )

        return run_id
    finally:
        cursor.close()


def run_with_transaction(
    results: List[Dict[str, Any]],
    *,
    model_name: Optional[str] = "gpt-4o",
    prompt_version: Optional[str] = None,
    pipeline_version: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Open connection, write_run in one transaction, commit. On exception, rollback and re-raise.
    Returns run_id.
    """
    import mysql.connector
    cfg = config or get_mysql_config()
    conn = mysql.connector.connect(**cfg)
    try:
        run_id = write_run(
            conn,
            results,
            model_name=model_name,
            prompt_version=prompt_version,
            pipeline_version=pipeline_version,
        )
        conn.commit()
        logger.info("Committed run_id=%s", run_id)
        return run_id
    except Exception as e:
        conn.rollback()
        logger.exception("Rolled back after error: %s", e)
        raise
    finally:
        conn.close()
