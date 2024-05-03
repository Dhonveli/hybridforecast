import inspect
import pdb
from typing import Any, Callable, Dict, Optional, Union

import pandas as pd

class HybridForecast:
    """
    Hybrid Forecasting model that decomposes time series into components and forecasts each component separately
    """

    def __init__(
        self,
        decomposition_model: Callable,
        trend_models: Callable,
        seasonal_models: Dict[
            str,
            Callable,
        ],
        error_models: Callable,
        decomposition_kwargs: Dict[str, Any] = {},
        trend_kwargs: Dict[str, Any] = {},
        seasonal_kwargs: Dict[str, Dict[str, Any]] = {},
        error_kwargs: Dict[str, Any] = {},
    ):
        """
        Args:
            decomposition_model: decomposition model (e.g. statsmodels.MSTL, statsmodels.STL)
            trend_models: trend models (e.g. MLForecast, StatsForecast, NeuralForecast)
            seasonal_models: seasonal models (e.g. MLForecast, StatsForecast, NeuralForecast)
            error_models: error models (e.g. MLForecast, StatsForecast, NeuralForecast)
            decomposition_kwargs: decomposition model kwargs
            trend_kwargs: trend model kwargs
            seasonal_kwargs: seasonal model kwargs
            error_kwargs: error model kwargs

        Returns:
            None
        """

        self.decomposition_model = decomposition_model
        self.trend_models = trend_models
        self.seasonal_models = seasonal_models
        self.error_models = error_models

        self.decomposition_kwargs = decomposition_kwargs
        self.trend_kwargs = trend_kwargs
        self.seasonal_kwargs = seasonal_kwargs
        self.error_kwargs = error_kwargs

    def _decomposition(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Decompose time series into components

        Args:
            df: input data

        Returns:
            pd.DataFrame: data frame with components
        """

        df_copy = df.copy()
        target = df["y"]
        target.index = df["ds"]

        res = self.decomposition_model(target, **self.decomposition_kwargs).fit()

        resid = res.resid
        trend = res.trend
        seasonal = res.seasonal

        components = pd.concat([resid, trend, seasonal], axis=1)
        components.reset_index(inplace=True)
        pdb.set_trace()

        if "period" in self.decomposition_kwargs:
            assert components.shape[1] - 3 == len(self.decomposition_kwargs["period"])
            components.columns = ["ds", "resid", "trend"] + [
                f"seasonal_{i}" for i in self.decomposition_kwargs["period"]
            ]
        elif "periods" in self.decomposition_kwargs:
            assert components.shape[1] - 3 == len(self.decomposition_kwargs["periods"])
            components.columns = ["ds", "resid", "trend"] + [
                f"seasonal_{i}" for i in self.decomposition_kwargs["periods"]
            ]
        else:
            assert components.shape[1] - 3 == 1
            components.columns = ["ds", "resid", "trend", "seasonal"]

        # rename columns based on seasonal_{period}
        return pd.merge(
            df_copy[["y", "ds", "unique_id"]], components, on="ds", how="inner"
        )

    def _fit_model(
        self, df: pd.DataFrame, model, static_features=[], dynamic_features=[]
    ) -> Any:
        """
        Fit model

        Args:
            df: input data
            model: model
            static_features: static features
            dynamic_features: dynamic features

        Returns:
            model: fitted model

        """

        df = df[["unique_id", "ds", "y"] + dynamic_features + static_features]

        fit_args = inspect.signature(model.fit).parameters.keys()

        if static_features == [] and "static_features" in fit_args:
            model.fit(df, static_features=static_features)
        else:
            model.fit(df)

        return model

    def _pred_model(
        self,
        model,
        h: int,
        X_df: Optional[pd.DataFrame] = None,
        dynamic_features=None,
    ) -> pd.DataFrame:
        """
        Predict model

        Args:
            model: model
            h: forecast horizon
            X_df: exogenous features
            dynamic_features: dynamic features

        Returns:
            pd.DataFrame: predictions
        """

        if dynamic_features:
            X_df = X_df[["unique_id", "ds"] + dynamic_features]
            return model.predict(h=h, X_df=X_df)
        else:
            return model.predict(h=h)

    def _combine_predictions(
        self, predictions: Dict[str, pd.DataFrame], agg="additive"
    ) -> pd.DataFrame:
        """
        Combine predictions

        Args:
            predictions: dict with predictions
            agg: aggregation method

        Returns:
            pd.DataFrame: combined predictions
        """

        # combine predictions
        idx = 0
        combination_df = pd.DataFrame()

        # iterate over components
        for comb, prediction in predictions.items():

            # check if unique_id column is present
            if "unique_id" in prediction.columns:
                prediction_copy = prediction.copy()
            else:
                prediction_copy = prediction.reset_index().copy()

            model_cols = [
                col for col in prediction_copy.columns if col not in ["unique_id", "ds"]
            ]

            # init dataframes
            if idx == 0:
                prediction_df = prediction_copy[["unique_id", "ds"]]
                idx += 1
                for col in model_cols:
                    prediction_df[f"{comb}_{col}"] = prediction_copy[col]
                    combination_df = prediction_df.copy()
            else:
                combination_cols = [
                    col
                    for col in combination_df.columns
                    if col not in ["unique_id", "ds"]
                ]
                for col in model_cols:
                    prediction_df[f"{comb}_{col}"] = prediction_copy[col]

                    # combine predictions
                    for col_comb in combination_cols:
                        if agg == "additive":
                            combination_df[f"{col_comb}_{comb}_{col}"] = (
                                prediction_copy[col] + combination_df[col_comb]
                            )
                        elif agg == "multiplicative":
                            combination_df[f"{col_comb}_{comb}_{col}"] = (
                                prediction_copy[col] * combination_df[col_comb]
                            )
                combination_df.drop(columns=combination_cols, inplace=True)

        return prediction_df.merge(combination_df, on=["unique_id", "ds"])

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit model

        Args:
            df: input data

        Returns:
            None
        """

        self.components_df = self._decomposition(df)
        components_names = self.components_df.columns.drop(["ds", "unique_id", "y"])

        # forecast components in parallel (not trend)
        self.model_error_fitted = None
        self.model_trend_fitted = None
        self.model_seasonal_fitted = {}

        for component in components_names:
            print(f"fitting {component}")
            if component == "trend":
                df_trend = self.components_df[["unique_id", "ds", component]].rename(
                    columns={component: "y"}
                )

                # get list exogenous features for trend
                dynamic_features = self.trend_kwargs.get("dynamic_features", [])
                static_features = self.trend_kwargs.get("static_features", [])

                # create df with exogenous features
                df_copy = df[
                    static_features + dynamic_features + ["unique_id", "ds"]
                ].copy()
                df_trend = df_trend.merge(df_copy, on=["unique_id", "ds"])

                self.model_trend_fitted = self._fit_model(
                    df_trend,
                    self.trend_models,
                    static_features=static_features,
                    dynamic_features=dynamic_features,
                )

            elif component == "resid":

                df_resid = self.components_df[["unique_id", "ds", component]].rename(
                    columns={component: "y"}
                )

                # get list exogenous features for resid
                dynamic_features = self.error_kwargs.get("dynamic_features", [])
                static_features = self.error_kwargs.get("static_features", [])

                # create df with exogenous features
                df_copy = df[
                    static_features + dynamic_features + ["unique_id", "ds"]
                ].copy()
                df_resid = df_resid.merge(df_copy, on=["unique_id", "ds"])

                self.model_error_fitted = self._fit_model(
                    df_resid,
                    self.error_models,
                    static_features=static_features,
                    dynamic_features=dynamic_features,
                )

            elif "seasonal" in component:

                df_seasonal = self.components_df[["unique_id", "ds", component]].rename(
                    columns={component: "y"}
                )

                # get list exogenous features for seasonal
                dynamic_features = self.seasonal_kwargs.get("dynamic_features", [])
                static_features = self.seasonal_kwargs.get("static_features", [])

                # create df with exogenous features
                df_copy = df[
                    static_features + dynamic_features + ["unique_id", "ds"]
                ].copy()
                df_seasonal = df_seasonal.merge(df_copy, on=["unique_id", "ds"])

                self.model_seasonal_fitted[component] = self._fit_model(
                    df_seasonal,
                    self.seasonal_models[component],
                    static_features=static_features,
                    dynamic_features=dynamic_features,
                )

            else:
                raise ValueError(f" {component} not recognized")

    def predict(self, h: int, X_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Predict model

        Args:
            h: forecast horizon
            X_df: exogenous features

        Returns:
            pd.DataFrame: predictions
        """

        components_names = self.components_df.columns.drop(["ds", "unique_id", "y"])

        predictions = {}
        # forecast components in parallel (not trend)
        for component in components_names:
            print(f"predicting {component}")
            if component == "trend":
                predictions[component] = self._pred_model(
                    self.model_trend_fitted,
                    h=h,
                    X_df=X_df,
                    dynamic_features=self.trend_kwargs.get("dynamic_features", None),
                )
            elif component == "resid":
                predictions[component] = self._pred_model(
                    self.model_error_fitted,
                    h=h,
                    X_df=X_df,
                    dynamic_features=self.error_kwargs.get("dynamic_features", None),
                )

            elif "seasonal" in component:
                predictions[component] = self._pred_model(
                    self.model_seasonal_fitted[component],
                    h=h,
                    X_df=X_df,
                    dynamic_features=self.seasonal_kwargs.get("dynamic_features", None),
                )

        return self._combine_predictions(predictions)
