from sqlmodel import Session
from apps.api.core.db import engine
from sqlalchemy import text

with Session(engine) as s:
    rows = s.exec(text("""
        select column_name
        from information_schema.columns
        where table_name = 'users'
        order by ordinal_position
    """)).all()
    print(rows)
