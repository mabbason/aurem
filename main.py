"""
Aurem — Entry point.
Runs entirely on Windows: audio capture + transcription + web UI in one process.
"""

import asyncio
import uvicorn

from server.app import app, pipeline


async def main():
    print("=" * 50)
    print("  Aurem")
    print("=" * 50)
    print()

    # Load ML models (Whisper + optional diarization)
    pipeline.load_models()
    print()

    # Start web server
    uv_config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8765,
        log_level="info",
    )
    server = uvicorn.Server(uv_config)

    print("Web UI: http://localhost:8765")
    print("Click 'Start Recording' in the browser to begin.")
    print()

    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
