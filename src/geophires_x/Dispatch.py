from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from geophires_x.OptionList import DispatchDemandSource, DispatchFlowStrategy, EndUseOptions, OperatingMode
from geophires_x.OptionList import PlantType

if TYPE_CHECKING:
    from geophires_x.Model import Model


def _enum_value_or_str(value: Any) -> str:
    return getattr(value, "value", value)


def _as_float_series(raw_series: Any) -> np.ndarray:
    series = np.asarray(raw_series, dtype=float)
    if series.ndim == 2:
        if series.shape[1] < 2:
            raise ValueError("Dispatch demand profile must contain time-value pairs when a 2D historical array is provided.")
        return series[:, 1]

    if series.ndim != 1:
        raise ValueError("Dispatch demand profile must be a one-dimensional series or a historical array of time-value pairs.")

    return series


@dataclass
class DemandProfile:
    series: np.ndarray
    units: str
    num_timesteps: int
    time_step_hours: float = 1.0


class DemandProfileFactory:
    @staticmethod
    def from_model(model: "Model") -> DemandProfile:
        source = model.surfaceplant.dispatch_demand_source.value
        if not isinstance(source, DispatchDemandSource):
            source = DispatchDemandSource.from_input_string(source)
        if source != DispatchDemandSource.ANNUAL_HEAT_DEMAND:
            raise ValueError(f"Unsupported dispatch demand source: {_enum_value_or_str(source)}")

        raw_series = model.surfaceplant.HeatingDemand.value
        if len(raw_series) == 0:
            raise ValueError("Dispatchable mode requires `Annual Heat Demand` to be provided as an hourly demand profile.")

        series = _as_float_series(raw_series)
        if len(series) != 8760:
            raise ValueError(
                f"Dispatchable mode requires an hourly one-year demand profile with 8760 timesteps; received {len(series)}."
            )

        units = _enum_value_or_str(getattr(model.surfaceplant.HeatingDemand, "CurrentYUnits", ""))
        return DemandProfile(series=series, units=str(units), num_timesteps=len(series))


@dataclass
class DispatchCommand:
    target_flow_fraction: float
    runtime_fraction: float
    is_shut_in: bool


@dataclass
class DispatchTimestepResult:
    produced_temperature: float = 0.0
    plant_outlet_thermal_power: float = 0.0
    pumping_power: float = 0.0
    electrical_output: float = 0.0
    served_thermal_demand: float = 0.0
    unmet_thermal_demand: float = 0.0
    actual_flow: float = 0.0
    runtime_fraction: float = 0.0


@dataclass
class DispatchResults:
    hourly_produced_temperature: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hourly_flow: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hourly_runtime_fraction: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hourly_demand_served: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hourly_unmet_demand: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hourly_pumping_power: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hourly_geothermal_thermal_output: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    annual_aggregates: dict[str, float] = field(default_factory=dict)
    summary_metrics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def initialize(cls, num_timesteps: int) -> "DispatchResults":
        zeros = np.zeros(num_timesteps, dtype=float)
        return cls(
            hourly_produced_temperature=zeros.copy(),
            hourly_flow=zeros.copy(),
            hourly_runtime_fraction=zeros.copy(),
            hourly_demand_served=zeros.copy(),
            hourly_unmet_demand=zeros.copy(),
            hourly_pumping_power=zeros.copy(),
            hourly_geothermal_thermal_output=zeros.copy(),
        )


class DispatchStrategy(ABC):
    @abstractmethod
    def dispatch(self, timestep_state: dict[str, Any], demand: float) -> DispatchCommand:
        raise NotImplementedError


class DemandFollowingDispatchStrategy(DispatchStrategy):
    def dispatch(self, timestep_state: dict[str, Any], demand: float) -> DispatchCommand:
        if demand <= 0:
            return DispatchCommand(target_flow_fraction=0.0, runtime_fraction=0.0, is_shut_in=True)

        max_flow_fraction = timestep_state.get("maximum_dispatch_flow_fraction", 1.0)
        runtime_fraction = 1.0
        return DispatchCommand(
            target_flow_fraction=min(max_flow_fraction, 1.0),
            runtime_fraction=runtime_fraction,
            is_shut_in=False,
        )


class DispatchPlantAdapter(ABC):
    @abstractmethod
    def initialize(self, model: "Model", design_state: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def evaluate_timestep(self, dispatch_command: DispatchCommand, timestep_index: int) -> DispatchTimestepResult:
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> None:
        raise NotImplementedError


class PlaceholderDispatchPlantAdapter(DispatchPlantAdapter):
    def __init__(self, supported_reservoir_name: str):
        self._supported_reservoir_name = supported_reservoir_name

    def initialize(self, model: "Model", design_state: dict[str, Any]) -> None:
        return None

    def evaluate_timestep(self, dispatch_command: DispatchCommand, timestep_index: int) -> DispatchTimestepResult:
        raise NotImplementedError(
            f"Dispatchable timestep simulation has not been implemented yet for {self._supported_reservoir_name}."
        )

    def finalize(self) -> None:
        return None


class DispatchAdapterFactory:
    _registry = {
        "CylindricalReservoir": lambda: PlaceholderDispatchPlantAdapter("CylindricalReservoir"),
        "MPFReservoir": lambda: PlaceholderDispatchPlantAdapter("MPFReservoir"),
        "LHSReservoir": lambda: PlaceholderDispatchPlantAdapter("LHSReservoir"),
        "SFReservoir": lambda: PlaceholderDispatchPlantAdapter("SFReservoir"),
        "UPPReservoir": lambda: PlaceholderDispatchPlantAdapter("UPPReservoir"),
        "SBTReservoir": lambda: PlaceholderDispatchPlantAdapter("SBTReservoir"),
    }

    @classmethod
    def create(cls, model: "Model") -> DispatchPlantAdapter:
        adapter_factory = cls._registry.get(model.reserv.__class__.__name__)
        if adapter_factory is not None:
            return adapter_factory()

        raise ValueError(
            f"Dispatchable mode is not supported for reservoir type `{model.reserv.__class__.__name__}`."
        )


class OperatingModeStrategy(ABC):
    @abstractmethod
    def run(self, model: "Model") -> None:
        raise NotImplementedError


class BaseloadOperatingModeStrategy(OperatingModeStrategy):
    def run(self, model: "Model") -> None:
        model.reserv.Calculate(model)
        model.wellbores.Calculate(model)
        model.surfaceplant.Calculate(model)

        if model.surfaceplant.plant_type.value == PlantType.DISTRICT_HEATING:
            model.reserv.Calculate(model)
            model.wellbores.Calculate(model)
            model.surfaceplant.Calculate(model)

        model.economics.Calculate(model)


class DispatchableOperatingModeStrategy(OperatingModeStrategy):
    def __init__(self, dispatch_strategy: DispatchStrategy | None = None):
        self._dispatch_strategy = dispatch_strategy or DemandFollowingDispatchStrategy()

    def run(self, model: "Model") -> None:
        if model.surfaceplant.enduse_option.value != EndUseOptions.HEAT:
            raise ValueError("Dispatchable mode currently supports only direct-use heat cases.")

        flow_strategy = model.surfaceplant.dispatch_flow_strategy.value
        if not isinstance(flow_strategy, DispatchFlowStrategy):
            flow_strategy = DispatchFlowStrategy.from_input_string(flow_strategy)

        if flow_strategy != DispatchFlowStrategy.DEMAND_FOLLOWING:
            raise ValueError(
                f"Unsupported dispatch flow strategy: {_enum_value_or_str(flow_strategy)}"
            )

        demand_profile = DemandProfileFactory.from_model(model)
        model.dispatch_results = DispatchResults.initialize(demand_profile.num_timesteps)
        model.dispatch_adapter = DispatchAdapterFactory.create(model)
        model.dispatch_adapter.initialize(model, design_state={})

        raise NotImplementedError(
            "Dispatchable operating mode framework is configured, but the hourly dispatch simulation loop is not "
            "implemented yet."
        )


def create_operating_mode_strategy(operating_mode: OperatingMode) -> OperatingModeStrategy:
    if operating_mode == OperatingMode.DISPATCHABLE:
        return DispatchableOperatingModeStrategy()

    return BaseloadOperatingModeStrategy()
