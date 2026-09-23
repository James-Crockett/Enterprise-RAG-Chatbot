from __future__ import annotations

from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID, uuid4

from sqlmodel import SQLModel, Field, Column
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    email: str = Field(nullable=False, unique=True, index=True)
    hashed_password: str

    max_access_level: int = Field(default=0, nullable=False)
    is_active: bool = Field(default=True, nullable=False)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class Document(SQLModel, table=True):
    __tablename__ = "documents"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    title: str
    source_path: Optional[str] = None
    department: str = Field(default="general", nullable=False)
    access_level: int = Field(default=1, nullable=False)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class Chunk(SQLModel, table=True):
    __tablename__ = "chunks"

    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: UUID = Field(foreign_key="documents.id")
    chunk_index: int
    page: Optional[int] = None
    text: str

    # sqlalchemy reserves metadata, so the model uses meta for that column.
    meta: Dict[str, Any] = Field(
        sa_column=Column("metadata", JSONB, nullable=False, default={})
    )

    access_level: int = Field(default=1, nullable=False)
    embedding: List[float] = Field(sa_column=Column(Vector(384), nullable=False))


class Conversation(SQLModel, table=True):
    __tablename__ = "conversations"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="users.id", index=True)
    title: str
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class Message(SQLModel, table=True):
    __tablename__ = "messages"

    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: UUID = Field(foreign_key="conversations.id", index=True)
    role: str
    content: str
    mode: Optional[str] = None

    # snapshot of the cited chunks so old answers keep their sources.
    sources: List[Dict[str, Any]] = Field(
        sa_column=Column(JSONB, nullable=False, default=[])
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
