"""Import LightGBM before anything gets a chance to import PyTorch.

Both LightGBM (via bugflow_ml's commit-risk model) and PyTorch (via
sentence-transformers, used for defect duplicate detection) bundle their own
copy of the OpenMP runtime. Whichever loads first in a process determines
which runtime "wins"; loading torch first and LightGBM second segfaults
(reproduced directly: `import sentence_transformers; ...; import lightgbm`
crashes with SIGSEGV, while the reverse order doesn't). Since a single API or
worker process legitimately serves both commit-risk and defect-duplicate
requests, the crash is a real ordering risk, not just a test artifact.

Import this module first thing in every process entrypoint — the API
(`app.main`), the RQ worker (`app.workers.run`), and the test suite
(`tests.conftest`) — before any request/job/test can import
sentence_transformers first.
"""

import lightgbm  # noqa: F401
