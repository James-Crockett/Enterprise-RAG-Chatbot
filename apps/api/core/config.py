import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://rag:rag@127.0.0.1:5432/rag_kb",
)

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALG = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "120"))
