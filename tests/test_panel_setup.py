import unittest

import numpy as np
import pandas as pd

from paneltime.main import Input
from paneltime.options import FitOptions, OptimizerOptions
from paneltime.processing.model_parser import (
    concatenate,
    eval_variables,
    extend_timeseries,
)
from paneltime.processing.panel import Panel


class PanelSetupTests(unittest.TestCase):
    def test_existing_columns_preserve_order_dtypes_and_missing_value_counts(self):
        index = pd.MultiIndex.from_product(
            [["a", "b"], range(3)], names=["entity", "date"]
        )
        data = pd.DataFrame({
            "integer": np.arange(6),
            "value": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
        }, index=index)
        data.columns.name = "variables"
        for dropna in (True, False):
            with self.subTest(dropna=dropna):
                result, lost, lags, history, size = eval_variables(
                    data, ["value", "integer", "value"], ["entity"], dropna
                )
                expected = data.dropna() if dropna else data
                expected = expected[["value", "integer"]].rename_axis(columns=None)
                pd.testing.assert_frame_equal(result, expected)
                self.assertEqual((lost, lags, history, size),
                                 (int(dropna), 0, 0, 6))
                result.iloc[0, 0] = -100
                self.assertEqual(data.iloc[0, 1], 1.0)

    def test_expression_evaluation_retains_lag_history_and_missing_values(self):
        index = pd.MultiIndex.from_product(
            [["a"], range(5)], names=["entity", "date"]
        )
        data = pd.DataFrame({"x": np.arange(1.0, 6.0)}, index=index)
        result, lost, lags, history, size = eval_variables(
            data, ["x", "L(x,2)", "D(x)", "np.abs(x)"], ["entity"]
        )
        expected = pd.DataFrame({
            "x": data.x, "L(x,2)": data.x.shift(2),
            "D(x)": data.x.diff(), "np.abs(x)": data.x.abs(),
        })
        pd.testing.assert_frame_equal(result, expected)
        self.assertEqual((lost, lags, history, size), (0, 2, 2, 5))

    def test_nullable_forecast_concatenation(self):
        index = pd.MultiIndex.from_product(
            [["a"], [0, 1]], names=["entity", "date"]
        )
        data = pd.DataFrame({
            "integer": [1, 2], "value": [1.5, np.nan], "text": ["a", None],
        }, index=index)
        future = pd.DataFrame(
            pd.NA,
            index=pd.MultiIndex.from_tuples([("a", 2)], names=index.names),
            columns=data.columns,
        )
        original = data.copy(deep=True)
        result = concatenate(future, data)
        expected = pd.DataFrame({
            "integer": pd.array([1, 2, pd.NA], dtype="Int64"),
            "value": pd.array([1.5, pd.NA, pd.NA], dtype="Float64"),
            "text": pd.array(["a", pd.NA, pd.NA], dtype=object),
        }, index=pd.MultiIndex.from_product(
            [["a"], [0, 1, 2]], names=index.names
        ))
        pd.testing.assert_frame_equal(result, expected)
        pd.testing.assert_frame_equal(data, original)
        self.assertTrue(future.isna().all().all())

    def test_forecast_rows_only_extend_groups_reaching_the_last_date(self):
        for dated in (False, True):
            for lags in (0, 3):
                with self.subTest(dated=dated, lags=lags):
                    delta = pd.Timedelta(days=1) if dated else 1
                    start = pd.Timestamp("2020-01-01") if dated else 0
                    tuples = [
                        (entity, start + time * delta)
                        for entity, count in (("a", 4), ("b", 3), ("c", 4))
                        for time in range(count)
                    ]
                    index = pd.MultiIndex.from_tuples(
                        tuples, names=["entity_orig", "date_orig"]
                    )
                    data = pd.DataFrame({
                        "entity": [0] * 4 + [1] * 3 + [2] * 4,
                        "date": [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3],
                        "x": np.arange(11, dtype=float),
                    }, index=index)
                    for name in ("Intercept", "var_const", "instrument_intercept", "one"):
                        data[name] = 1
                    result, pred_start = extend_timeseries(
                        data, ["entity_orig"], ["entity"], ["date_orig"],
                        ["date"], 1, delta, lags, 2,
                    )
                    self.assertEqual(pred_start, start + 4 * delta)
                    future = result.loc[
                        result.index.get_level_values("date_orig") >= pred_start
                    ]
                    horizon = max(lags, 1)
                    expected_index = pd.MultiIndex.from_product(
                        [["a", "c"], [start + t * delta for t in range(4, 4 + horizon)]],
                        names=index.names,
                    )
                    pd.testing.assert_index_equal(future.index, expected_index)
                    np.testing.assert_array_equal(
                        future.entity.to_numpy(), np.repeat([0, 2], horizon)
                    )
                    np.testing.assert_array_equal(
                        future.date.to_numpy(), np.tile(np.arange(4, 4 + horizon), 2)
                    )
                    self.assertTrue(future.x.isna().all())
                    self.assertTrue((future[["Intercept", "var_const",
                                             "instrument_intercept", "one"]] == 1).all().all())
                    history = result.loc[
                        result.index.get_level_values("date_orig") < pred_start
                    ]
                    self.assertEqual(set(history.index.get_level_values(0)), {"a", "c"})
                    np.testing.assert_array_equal(history.date.to_numpy(), [1, 2, 3, 1, 2, 3])

    def test_group_arrays_and_calendar_maps_match_observed_rows(self):
        rng = np.random.default_rng(42)
        rows = [(entity, date, rng.normal(), rng.normal())
                for entity, dates in (
                    (9, [0, 1, 3, 4, 6]), (2, [1, 2, 4, 6]),
                    (7, [0, 3, 5]), (4, [99]),
                ) for date in dates]
        data = pd.DataFrame(rows, columns=["id", "time", "x", "y"])
        data = data.sample(frac=1, random_state=3)
        original = data.copy(deep=True)
        options = FitOptions(
            order=(0, 0, 0), garch_order=(0, 0),
            optimizer=OptimizerOptions(min_group_df=2),
        ).to_engine_options()
        ip = Input(data, "y ~ x", "id", "time", options, None, None)
        panel = Panel(ip, options)
        panel.init()
        np.testing.assert_array_equal(panel.original_names, [2, 7, 9])
        for row, entity in enumerate(panel.original_names):
            selection = ip.idvar.iloc[:, 0].to_numpy() == entity
            count = int(selection.sum())
            np.testing.assert_array_equal(panel.X[row, :count], ip.X.to_numpy()[selection])
            np.testing.assert_array_equal(panel.Y[row, :count], ip.Y.to_numpy()[selection])
            np.testing.assert_array_equal(panel.map[row, :count], np.flatnonzero(selection))
            self.assertFalse(panel.included[3][row, count:].any())
            self.assertFalse(panel.X[row, count:].any())
        observed_dates = []
        for coordinates, count in zip(panel.date_map, panel.date_count[:, 0, 0]):
            dates = panel.timevar[coordinates][:, 0]
            self.assertTrue((dates == dates[0]).all())
            self.assertEqual(len(dates), count)
            np.testing.assert_array_equal(panel.date_count_mtrx[coordinates], count)
            observed_dates.append(dates[0])
        np.testing.assert_array_equal(observed_dates, np.arange(7))
        pd.testing.assert_frame_equal(data, original)

    def test_prediction_arrays_preserve_missing_groups_and_input_frames(self):
        rng = np.random.default_rng(12)
        for dated in (False, True):
            for lagged in (False, True):
                with self.subTest(dated=dated, lagged=lagged):
                    rows = [
                        (entity, time, rng.normal(), rng.normal())
                        for entity, count in (("a", 8), ("b", 7), ("c", 8))
                        for time in range(count)
                    ]
                    data = pd.DataFrame(rows, columns=["id", "time", "x", "y"])
                    if dated:
                        data["time"] = (
                            pd.Timestamp("2020-01-01")
                            + pd.to_timedelta(data.time, unit="D")
                        )
                    options = FitOptions(
                        order=(1, 0, 0) if lagged else (0, 0, 0),
                        garch_order=(0, 0),
                    ).to_engine_options()
                    formula = "y ~ L(x,2) + x" if lagged else "y ~ x"
                    ip = Input(data, formula, "id", "time", options, None, None)
                    original = ip.X_pred.copy(deep=True)
                    panel = Panel(ip, options)
                    panel.init()
                    horizon = 2 if lagged else 1
                    self.assertEqual(panel.X_pred.shape, (3, horizon, len(ip.X.columns)))
                    np.testing.assert_array_equal(panel.X_is_predicted, [True, False, True])
                    self.assertFalse(panel.X_pred[1].any())
                    self.assertTrue(np.isnan(panel.X_pred_timevar[1]).all())
                    self.assertTrue(pd.isna(panel.X_pred_idvar[1]).all())
                    for row, entity in ((0, "a"), (2, "c")):
                        code = ip.idvar.loc[ip.idvar_orig.iloc[:, 0] == entity].iloc[0, 0]
                        selection = ip.idvar_pred.iloc[:, 0].to_numpy() == code
                        expected = original.loc[selection].to_numpy(
                            dtype=float, na_value=np.nan
                        )
                        np.testing.assert_array_equal(panel.X_pred[row], expected)
                        np.testing.assert_array_equal(panel.X_pred_idvar[row, :, 0], entity)
                    pd.testing.assert_frame_equal(ip.X_pred, original)


if __name__ == "__main__":
    unittest.main()
