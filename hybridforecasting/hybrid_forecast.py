import inspect
from typing import Any, Callable, Dict, Optional, Union

import pandas as pd

import pdb
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd


class HybridForecast:
    """
    Hybrid Forecasting model that decomposes time series into components and forecasts each component separately

    Attributes:
        decomposition_model (Callable): decomposition model (e.g. statsmodels.MSTL, statsmodels.STL)
        trend_models (List[Callable]): trend models (e.g. MLForecast, StatsForecast, NeuralForecast)
        seasonal_models (Dict[str, Callable]): seasonal models (e.g. MLForecast, StatsForecast, NeuralForecast)
        resid_models (List[Callable]): resid models (e.g. MLForecast, StatsForecast, NeuralForecast)
        decomposition_kwargs (Dict[str, Any]): decomposition model kwargs
        trend_kwargs (Dict[str, Any]): trend model kwargs
        seasonal_kwargs (Dict[str, Dict[str, Any]]): seasonal model kwargs
        resid_kwargs (Dict[str, Any]): resid model kwargs
    """

    def __init__(
        self,
        decomposition_model: Callable,
        trend_model: Callable,
        seasonal_model: Dict[str, Callable],
        resid_model: Callable,
        decomposition_kwargs: Dict[str, Any] = {},
        trend_kwargs: Dict[str, Any] = {},
        seasonal_kwargs: Dict[str, Dict[str, Any]] = {},
        resid_kwargs: Dict[str, Any] = {},
        agg: str = "additive",
    ):
        """
        Initialize HybridForecast object

        Args:
            decomposition_model (Callable): decomposition model
            trend_models (List[Callable]): trend models
            seasonal_models (Dict[str, Callable]): seasonal models
            resid_models (List[Callable]): resid models
            decomposition_kwargs (Dict[str, Any], optional): decomposition model kwargs. Defaults to {}.
            trend_kwargs (Dict[str, Any], optional): trend model kwargs. Defaults to {}.
            seasonal_kwargs (Dict[str, Dict[str, Any]], optional): seasonal model kwargs. Defaults to {}.
            resid_kwargs (Dict[str, Any], optional): resid model kwargs. Defaults to {}.
            agg (str, optional): aggregation function. Defaults to "additive".
        """

        self.decomposition_model = decomposition_model
        self.trend_model = trend_model
        self.seasonal_model = seasonal_model
        self.resid_model = resid_model

        self.decomposition_kwargs = decomposition_kwargs
        self.trend_kwargs = trend_kwargs
        self.seasonal_kwargs = seasonal_kwargs
        self.resid_kwargs = resid_kwargs
        self.agg = agg

    def _decompose_ts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Decompose time series into components

        Args:
            df (pd.DataFrame): input data

        Returns:
            pd.DataFrame: data frame with components
        """
        
        components = pd.DataFrame()
        
        for id in df["unique_id"].unique():
            
            df_temp = df[df["unique_id"] == id]
            df_temp = df_temp.sort_values(by="ds")
            
            target = df_temp["y"]
            target.index = df_temp["ds"]

            res = self.decomposition_model(target, **self.decomposition_kwargs).fit()

            if len(res.seasonal.shape) == 1:
                component_tmp = pd.concat(
                    [res.resid, res.trend, res.seasonal], axis=1, keys=["resid", "trend", "seasonal"]
                )
            else:
                components_tmp = pd.concat(
                    [res.resid, res.trend],
                    axis=1,
                    keys=["resid", "trend"],
                )
                
                components_tmp = pd.merge(
                    components_tmp, res.seasonal, left_index=True, right_index=True
                )

            components_tmp.reset_index(inplace=True)
            components_tmp["unique_id"] = id
        
            components = pd.concat([components, components_tmp])
            
        return pd.merge(df[["y", "ds", "unique_id"]], components, on=["ds", "unique_id"], how="inner")

    def _fit_model(
        self,
        df: pd.DataFrame,
        model:Callable,
        static_features: List[str] = [],
        dynamic_features: List[str] = [],
    ) -> Any:
        """
        Fit model

        Args:
            df (pd.DataFrame): input data
            model (Callable): model
            static_features (List[str], optional): static features. Defaults to [].
            dynamic_features (List[str], optional): dynamic features. Defaults to [].

        Returns:
            Any: fitted model
        """

        df = df[["unique_id", "ds", "y"] + dynamic_features + static_features]

        fit_args = inspect.signature(model.fit).parameters.keys()

        if static_features == [] and "static_features" in fit_args:
            model.fit(df, static_features=static_features)
        else:
            model.fit(df)

        return model

    def _predict_model(
        self,
        model: Callable,
        h: int,
        X_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Predict model

        Args:
            model (Callable): model
            h (int): forecast horizon
            X_df (Optional[pd.DataFrame], optional): exogenous features. Defaults to None.

        Returns:
            pd.DataFrame: predictions
        """

        if X_df.shape[1] > 2:
            return model.predict(h=h, X_df=X_df)
        else:
            return model.predict(h=h)

    def _combine_predictions(
        self, predictions: Dict[str, pd.DataFrame], aggregation: str = "additive"
    ) -> pd.DataFrame:
        """
        Combine predictions by component.

        Args:
            predictions (Dict[str, pd.DataFrame]): Dictionary of component predictions.
            aggregation (str, optional): Aggregation method. Defaults to "additive".

        Returns:
            pd.DataFrame: Combined predictions.
        """

        combined_df = pd.DataFrame()
        
        idx = 0

        for component, prediction in predictions.items():
            if 'unique_id' in prediction.columns:
                prediction_df = prediction.reset_index(drop=True).copy()
            else:
                prediction_df = prediction.reset_index(drop=False).copy()

            prediction_df.rename(columns={col : f"{component}_{col}" for col in prediction_df.columns if col not in ["unique_id", "ds"]}, inplace=True)
            model_columns = [
                col for col in prediction_df.columns if col not in ["unique_id", "ds"]
            ]
            if idx == 0:
                combined_df = prediction_df.copy()
                old_combined_columns = [col for col in combined_df.columns if col not in ["unique_id", "ds"]]
                idx += 1
            else:
                combined_df = pd.merge(combined_df, prediction_df, on=["unique_id", "ds"], how="outer")
                old_combined_columns_new = []
                for column in model_columns:
                    for combined_column in old_combined_columns:
                        if aggregation == "additive":
                            combined_df[f"{combined_column}_{column}"] = (
                                combined_df[combined_column] + prediction_df[column]
                            )
                        elif aggregation == "multiplicative":
                            combined_df[f"{combined_column}_{column}"] = (
                                combined_df[combined_column] *prediction_df[column]
                            )
                        old_combined_columns_new.append(f"{combined_column}_{column}")
                if idx != 1:
                    combined_df.drop(columns=old_combined_columns, inplace=True)
                idx += 1
                old_combined_columns = old_combined_columns_new     

        return combined_df

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the model to the data.

        Args:
            data (pd.DataFrame): Input data.

        Returns:
            None
        """

        self.components_df = self._decompose_ts(data)
        components = self.components_df.columns.drop(["ds", "unique_id", "y"])

        self.fitted_models : Dict[str, Dict[str, Any]] = {
            "trend": {},
            "seasonal": {},
            "resid": {}
        }

        for component in components:
            df = self.components_df[["unique_id", "ds", component]].rename(
                columns={component: "y"}
            )
            dynamic_features = self.get_kwargs(component, "dynamic_features")
            static_features = self.get_kwargs(component, "static_features")
            df_copy = data[static_features + dynamic_features + ["unique_id", "ds"]]

            df = df.merge(df_copy, on=["unique_id", "ds"], how="left")

            self.fitted_models[component] = self._fit_model(
                df,
                self.get_model(component),
                static_features,
                dynamic_features,
            )

    def get_kwargs(self, component, key):
        """Get the value of a given key from the kwargs dictionary based on the component."""
        return self.get_model_kwargs(component).get(key, [])

    def get_model(self, component):
        """Get the model based on the component."""
        return (
            self.trend_model
            if component == "trend"
            else self.seasonal_model[component]
            if component.startswith("seasonal")
            else self.resid_model
            if component == "resid"
            else None
        )

    def get_model_kwargs(self, component):
        """Get the kwargs dictionary based on the component."""
        return (
            self.trend_kwargs
            if component == "trend"
            else self.seasonal_kwargs.get(component, {})
            if component.startswith("seasonal")
            else self.resid_kwargs
            if component == "resid"
            else None
        )

    def predict(self, h: int, X_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Predicts the model's future values based on the given forecast horizon.

        Args:
            h (int): The number of time steps to forecast into the future.
            X_df (Optional[pd.DataFrame], optional): exogenous features. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the predicted values for the trend, seasonal components, and residuals.
        """

        predictions = {}
        dynamic_features = {}

        # Predict trend and seasonal components
        for component in ["trend", *self.seasonal_model.keys(), "resid"]:
            model_kwargs = self.get_model_kwargs(component)
            dynamic_features[component] = model_kwargs.get("dynamic_features", [])
            predictions[component] = self._predict_model(
                model = self.fitted_models[component],
                h = h,
                X_df=X_df[["unique_id", "ds"] + dynamic_features[component]].reset_index(drop=True),
            )

        return self._combine_predictions(predictions, aggregation=self.agg)
