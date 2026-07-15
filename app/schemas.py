"""Pydantic request/response schemas for the FlightRisk API."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class FlightInput(BaseModel):
    """A single scheduled flight, described using only pre-flight information."""

    airline: str = Field(..., description="Operating/reporting carrier code, e.g. 'DL'.", examples=["DL"])
    origin: str = Field(..., description="Origin airport IATA code, e.g. 'JFK'.", examples=["JFK"])
    destination: str = Field(..., description="Destination airport IATA code, e.g. 'LAX'.", examples=["LAX"])
    month: int = Field(..., ge=1, le=12, description="Scheduled month (1-12).")
    day_of_week: int = Field(..., ge=1, le=7, description="Scheduled day of week (1=Mon ... 7=Sun).")
    crs_dep_time: int = Field(..., ge=0, le=2400, description="Scheduled departure time, HHMM format. 2400 is accepted as midnight.")
    crs_arr_time: int = Field(..., ge=0, le=2400, description="Scheduled arrival time, HHMM format. 2400 is accepted as midnight.")
    crs_elapsed_time: int = Field(..., gt=0, description="Scheduled flight duration in minutes.")
    distance: float = Field(..., gt=0, description="Great-circle distance in miles.")
    flight_date: str | None = Field(default=None, description="Optional ISO scheduled date for exact calendar features.")

    @field_validator("crs_dep_time", "crs_arr_time")
    @classmethod
    def _valid_hhmm(cls, v: int) -> int:
        if v == 2400:
            return v
        hour, minute = divmod(v, 100)
        if hour > 23 or minute > 59:
            raise ValueError("Time must use valid HHMM format, e.g. 1830 or 2400 for midnight.")
        return v

    @field_validator("airline", "origin", "destination")
    @classmethod
    def _upper_strip(cls, v: str) -> str:
        return v.strip().upper()

    model_config = {
        "json_schema_extra": {
            "example": {
                "airline": "DL",
                "origin": "JFK",
                "destination": "LAX",
                "month": 7,
                "day_of_week": 5,
                "crs_dep_time": 1830,
                "crs_arr_time": 2145,
                "crs_elapsed_time": 375,
                "distance": 2475,
                "flight_date": "2026-07-18",
            }
        }
    }


class EuropeanFlightInput(BaseModel):
    """Experimental European flight layer. Distance is auto-estimated if missing."""

    airline: str = Field(..., description="European airline code, e.g. 'IB'.")
    origin: str = Field(..., description="European origin airport IATA code, e.g. 'BCN'.")
    destination: str = Field(..., description="European destination airport IATA code, e.g. 'AMS'.")
    month: int = Field(..., ge=1, le=12)
    day_of_week: int = Field(..., ge=1, le=7)
    crs_dep_time: int = Field(..., ge=0, le=2400)
    crs_arr_time: int = Field(..., ge=0, le=2400)
    crs_elapsed_time: int = Field(..., gt=0)
    distance: float | None = Field(default=None, gt=0, description="Optional manual distance in miles.")

    @field_validator("crs_dep_time", "crs_arr_time")
    @classmethod
    def _valid_hhmm(cls, v: int) -> int:
        if v == 2400:
            return v
        hour, minute = divmod(v, 100)
        if hour > 23 or minute > 59:
            raise ValueError("Time must use valid HHMM format, e.g. 1830 or 2400 for midnight.")
        return v

    @field_validator("airline", "origin", "destination")
    @classmethod
    def _upper_strip(cls, v: str) -> str:
        return v.strip().upper()


class EuropeanContextOutput(BaseModel):
    status: str
    source: str
    year: int | None = None
    month: int | None = None
    airline: str | None = None
    origin: str | None = None
    destination: str | None = None
    airport_pair: str | None = None
    avg_arrival_delay_min: float | None = None
    pct_flights_15min_late: float | None = None
    cancelled_pct: float | None = None
    number_flights_matched: int | None = None
    matched_level: str


class BatchFlightInput(BaseModel):
    flights: list[FlightInput] = Field(..., min_length=1, max_length=500)


class LocalContributionOutput(BaseModel):
    feature: str
    contribution: float
    direction: str
    magnitude: float
    active_category: str | None = None
    raw_value: str | int | float | bool | None = None


class FlightReportInput(FlightInput):
    language: str = Field(default="en", pattern="^(en|es)$")
    flight_number: str | None = Field(default=None, max_length=12)
    flight_date: str | None = None


class ScheduleReportInput(BatchFlightInput):
    language: str = Field(default="en", pattern="^(en|es)$")



class PredictionOutput(BaseModel):
    delay_probability: float = Field(..., description="Post-hoc calibrated estimate of a 15+ minute arrival delay.")
    raw_model_score: float = Field(..., description="Uncalibrated score emitted by the selected classifier.")
    calibration_method: str = Field(..., description="Post-hoc calibration method applied to the raw score.")
    risk_level: str = Field(..., description="'low', 'moderate', or 'high'.")
    decision_threshold: float = Field(..., description="Probability threshold used for binary risk decisions.")
    top_factors: list[str] = Field(..., description="Non-causal schedule-context signals for this flight.")
    local_contributions: list[LocalContributionOutput] = Field(default_factory=list, description="Signed local pre-calibration log-odds contributions from the selected model.")
    explanation_scale: str = Field(default="log_odds_before_calibration", description="Scale used by local model contributions.")
    review_recommended: bool = Field(default=False, description="Whether the configured operational policy sends this flight to priority review.")
    operational_action: str = Field(default="standard_monitoring")
    policy_name: str = Field(default="fixed_threshold")
    policy_probability_cutoff: float | None = None
    policy_capacity_fraction: float | None = None


class EuropeanPredictionOutput(PredictionOutput):
    region: str
    airline_name: str
    origin_label: str
    destination_label: str
    distance_miles: float
    distance_source: str
    european_context: EuropeanContextOutput
    experimental: bool = True
    transfer_note: str


class BatchPredictionOutput(BaseModel):
    predictions: list[PredictionOutput]


class RankedPredictionOutput(PredictionOutput):
    rank: int
    risk_percentile: float
    risk_bucket: str


class RankingOutput(BaseModel):
    flights_ranked: int
    top_5pct_count: int
    top_10pct_count: int
    policy_capacity_count: int = 0
    policy_capacity_fraction: float = 0.10
    policy_name: str = "top_10pct_capacity"
    ranking_metric_note: str
    ranked_predictions: list[RankedPredictionOutput]


class LivenessResponse(BaseModel):
    status: str
    version: str


class ReadinessResponse(BaseModel):
    status: str
    version: str
    checks: dict[str, bool]
    model: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    trained_at_utc: str | None = None
    version: str | None = None
    release_name: str | None = None
    artifact_schema_version: str | None = None
    calibration_method: str | None = None
    historical_encoding: str | None = None
    n_train_rows: int | None = None
    n_test_rows: int | None = None
    validation_rows: int | None = None
    feature_columns: list[str]
    decision_threshold: float | None = None
    operational_policy: dict = Field(default_factory=dict)
    metrics: dict


class ModelCardResponse(BaseModel):
    name: str
    version: str
    calibration_method: str
    historical_encoding: str
    task: str
    target: str
    intended_use: str
    not_intended_use: str
    selected_model: str
    candidate_models: list[str]
    decision_threshold: float
    operational_policy: dict = Field(default_factory=dict)
    main_metrics: dict
    baseline_metrics: dict
    leakage_controls: list[str]


class MonitoringSummaryResponse(BaseModel):
    total_predictions: int
    average_probability: float | None = None
    high_risk_share: float | None = None
    moderate_or_high_risk_share: float | None = None
    latest_prediction_utc: str | None = None
    model_name: str | None = None
    model_version: str | None = None


class DriftResponse(BaseModel):
    status: str
    max_psi: float | None = None
    features: dict


class RegionCatalogResponse(BaseModel):
    region: str
    airports: list[dict]
    airlines: list[dict]
    context_summary: dict | None = None


class EuropeanContextSummaryResponse(BaseModel):
    available: bool
    rows: int
    source_path: str
    is_sample: bool | None = None
    real_data: bool | None = None
    total_matched_flights: int | None = None
    airlines: list[str]
    routes: list[str]
    months: list[int] | None = None
    note: str | None = None
