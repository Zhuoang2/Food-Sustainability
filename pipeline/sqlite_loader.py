"""
Load menu batch from data/mydb_clean.sqlite.
Returns list of dicts: item_id, restaurant_id, item_name, description (optional).
"""
import os
import sqlite3
from typing import List, Dict, Any


def load_menu_batch(sqlite_path: str, limit: int) -> List[Dict[str, Any]]:
    """
    Load first `limit` menu entries from SQLite, ordered by restaurant_id, name.
    Assigns item_id via rowid for stability.
    """
    if not os.path.isfile(sqlite_path):
        raise FileNotFoundError(f"SQLite DB not found: {sqlite_path}")
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT rowid AS item_id, restaurant_id, name AS item_name, description
        FROM menus
        ORDER BY restaurant_id, name
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def build_menu_items_text(batch: List[Dict[str, Any]]) -> str:
    """
    Build the input string for the LLM (same format as notebook).
    Format: "1 item_id: 1; restaurant_id: 1; Item Name [description] 2 item_id: 2; ..."
    """
    parts = []
    for i, row in enumerate(batch, start=1):
        item_id = row["item_id"]
        restaurant_id = row["restaurant_id"]
        name = row["item_name"] or ""
        desc = (row.get("description") or "").strip()
        line = f"{i} item_id: {item_id}; restaurant_id: {restaurant_id}; {name}"
        if desc:
            line += f" {desc}"
        parts.append(line)
    return " ".join(parts)
