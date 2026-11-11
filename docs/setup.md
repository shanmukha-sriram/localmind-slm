# Setup

## Requirements
Windows 11, Python 3.10+, Ollama, 8+ GB RAM (16 GB recommended), 10+ GB free disk. Docker Desktop is optional.

## Local install
See the README. Check hardware with `python scripts\check_system.py`.

## Running
`streamlit run frontend/streamlit_app.py`, then open http://localhost:8501.

## Docker
- **Option A (recommended):** the container reaches the Ollama app on your host through
  `host.docker.internal:11434`.
- **Option B:** `--profile ollama` also runs Ollama in a container; set
  `LOCALMIND_OLLAMA_HOST=http://ollama:11434`. Models live in the `ollama_data` volume.
- GPU: containers do not automatically get GPU access. NVIDIA needs the Windows driver, the
  WSL2 backend and a Compose GPU reservation. Integrated Intel/AMD GPUs are not available to
  containers.
- Run official benchmarks natively. In Docker the RAM columns are `n/a`.

## Security
Bind Streamlit to loopback (default in Compose). Never publish Ollama's port to the internet:
it has no authentication. Do not commit `.env`.

## Cloud hosting (optional)
Needs a remote inference server, authentication in front of both services, and a private
network between them. Costs run continuously, prompts leave the user's machine, and laptop
benchmark numbers do not apply to the server.