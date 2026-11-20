"""Request/response contracts. Pydantic v2 rejects bad input with a 422 before
any preprocessing or model code runs."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChurnRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "tenure_months": 5,
                    "monthly_charges": 95.5,
                    "total_charges": 477.5,
                    "support_tickets": 4,
                    "contract_type": "month-to-month",
                }
            ]
        },
    )

    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(gt=0, le=500)
    total_charges: float = Field(ge=0)
    support_tickets: int = Field(ge=0, le=50)
    contract_type: Literal["month-to-month", "one-year", "two-year"]


class ChurnResponse(BaseModel):
    request_id: str
    prediction: Literal["Churn", "Retain"]
    confidence_score: float
    churn_probability: float
    risk_level: Literal["Low", "Medium", "High"]
    inference_latency_ms: float
    model_version: str
