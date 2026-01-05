from sqlmodel import SQLModel, Session, create_engine
from apps.api.core.config import DATABASE_URL

engine = create_engine(DATABASE_URL, echo=False)

def get_session():
    with Session(engine) as session:
        yield session
