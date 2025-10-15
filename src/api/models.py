"""Pydantic models for API requests and responses."""

from pydantic import BaseModel


class ChunkRequest(BaseModel):
    """Request model for semantic chunking."""

    document: str


class ChunkResponse(BaseModel):
    """Response model for semantic chunking."""

    chunks: list[str]
    num_chunks: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_name: str
