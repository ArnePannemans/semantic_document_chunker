"""
Simple FastAPI server for serving the semantic chunking model.

This is a straightforward single-request API for testing and UI integration.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import ChunkRequest, ChunkResponse, HealthResponse
from src.config import APIConfig
from src.core.predictor import SemanticChunker

config = APIConfig()

app = FastAPI(
    title="Semantic Chunking API",
    description="Simple API for semantic document chunking using Qwen3-14B with LoRA",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chunker: SemanticChunker | None = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global chunker

    adapter_path = config.inference.adapter_path
    if adapter_path:
        print(f"Loading SemanticChunker with model from: {adapter_path}")
    else:
        print(f"Loading SemanticChunker with base model: {config.inference.base_model}")

    chunker = SemanticChunker(
        adapter_path=adapter_path,
        config=config.inference,
    )

    print("Model loaded successfully")
    print("API ready for requests")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_name": config.inference.adapter_path or config.inference.base_model,
    }


@app.post("/v1/chunk", response_model=ChunkResponse)
async def chunk_document(request: ChunkRequest):
    """
    Perform semantic chunking on a document.

    Takes a single document and returns chunks.
    """
    if chunker is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        chunks = chunker.chunk_document(request.document)

        return {
            "chunks": chunks,
            "num_chunks": len(chunks),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chunking failed: {str(e)}") from e


if __name__ == "__main__":
    import uvicorn

    print("Starting Simple Semantic Chunking API Server...")
    print(f"Server will be available at: http://{config.host}:{config.port}")
    print(f"API docs at: http://{config.host}:{config.port}/docs")
    print("")

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level,
    )
