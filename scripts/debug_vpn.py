from sqlmodel import Session
from apps.api.core.db import engine
from sqlalchemy import text

SQL = """
select
  d.title,
  d.department,
  d.access_level,
  left(c.text, 120) as preview
from chunks c
join documents d on d.id = c.document_id
where lower(coalesce(d.source_path, '')) like '%vpn%'
   or lower(c.text) like '%vpn%'
limit 20;
"""

def main():
    with Session(engine) as s:
        rows = s.exec(text(SQL)).all()
        print("Rows:", len(rows))
        for r in rows:
            print(r)

if __name__ == "__main__":
    main()
