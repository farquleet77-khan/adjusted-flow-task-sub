"""The Forecasting Service must reproduce the delivered December CSV and reject bad inputs."""

import pandas as pd
import pytest

from project_paths import DEC_CSV, MODELS, OUTPUTS

pytestmark = pytest.mark.skipif(not (MODELS / "intervals.joblib").exists() or not DEC_CSV.exists(),
                                reason="run scripts 03 and 04 first")


def test_service_reproduces_december_csv() -> None:
    from forecasting.service import predict
    out = predict(pd.read_csv(DEC_CSV)).round(2)
    ref = pd.read_csv(OUTPUTS / "december_forecast.csv")
    assert len(out) == 2976
    for col in ["load_kW", "load_kW_lower", "load_kW_upper"]:
        assert (out[col].to_numpy() - ref[col].to_numpy()).__abs__().max() < 0.011


def test_service_rejects_bad_inputs() -> None:
    from forecasting.service import predict
    dec = pd.read_csv(DEC_CSV)
    with pytest.raises(ValueError):
        predict(dec.drop(columns=["plant_state"]))
    with pytest.raises(ValueError):
        predict(dec, quantiles=(0.2,))
