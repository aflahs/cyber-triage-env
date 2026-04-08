"""Re-export for compatibility."""
import os
from cyber_triage_env.server.app import app  # noqa: F401


def main():
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
