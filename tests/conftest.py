import os
import tempfile
from pathlib import Path

# Before the app is imported: tests use a throwaway database and no rate limit
# (several tests send 50+ requests in a second; the limiter has its own test).
os.environ["DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test.db")
os.environ["RATE_LIMIT_PER_MINUTE"] = "0"
os.environ.pop("API_KEY", None)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

VALID = {
    "tenure_months": 5,
    "monthly_charges": 95.5,
    "total_charges": 477.5,
    "support_tickets": 4,
    "contract_type": "month-to-month",
}


@pytest.fixture(scope="session")
def client():
    # One app instance for the whole session: the model loads once, like in production.
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_state(client):
    client.app.state.store.clear()
    client.app.state.limiter.reset()
