import pytest
from fastapi.testclient import TestClient

from app.main import app

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
def reset_buffers(client):
    client.app.state.recent.clear()
    client.app.state.drift_window.clear()
