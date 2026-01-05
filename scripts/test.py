from sqlmodel import Session
from apps.api.core.db import engine
from sqlalchemy import text

with Session(engine) as s:
    rows = s.exec(text("""
        from sqlmodel import Session; from apps.api.core.db import engine; from sqlalchemy import text; s=Session(engine); 
        print(s.exec(text(\"select metadata->>'department', count(*) from chunks group by 1 order by 2 desc
    """)).all()
    print(rows)
