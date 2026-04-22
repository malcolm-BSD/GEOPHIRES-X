from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from geophires_x.GeoPHIRESUtils import quantity
from geophires_x.OptionList import Configuration, DispatchDemandSource, DispatchFlowStrategy, EndUseOptions, OperatingMode
from geophires_x.OptionList import PlantType
from geophires_x.Units import EnergyUnit, PowerUnit

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


def _as_baseline_series(raw_series: Any, parameter_name: str) -> np.ndarray:
    series = np.asarray(raw_series, dtype=float)
    if series.ndim == 0:
        series = np.array([float(series)], dtype=float)
    else:
        series = np.ravel(series).astype(float, copy=False)

    if series.size == 0:
        raise ValueError(f"Dispatchable mode requires a non-empty `{parameter_name}` baseline profile.")

    return series


def _coerce_series_length(
    raw_series: Any,
    target_length: int,
    parameter_name: str,
    default_value: float = 0.0,
) -> np.ndarray:
    series = np.asarray(raw_series, dtype=float)
    if series.ndim == 0:
        return np.full(target_length, float(series), dtype=float)

    series = np.ravel(series).astype(float, copy=False)
    if series.size == 0:
        return np.full(target_length, default_value, dtype=float)

    if series.size == 1 and target_length > 1:
        return np.full(target_length, float(series[0]), dtype=float)

    if series.size != target_length:
        raise ValueError(
            f"Dispatchable mode expected `{parameter_name}` to have {target_length} values, received {series.size}."
        )

    return series


@dataclass
class DemandProfile:
    series: np.ndarray
    units: str
    num_timesteps: int
    time_step_hours: float = 1.0


def _series_to_mw(series: np.ndarray, units: Any, time_step_hours: float) -> np.ndarray:
    units_value = _enum_value_or_str(units)

    if units in [PowerUnit.W, PowerUnit.KW, PowerUnit.MW, PowerUnit.GW] or units_value in [it.value for it in PowerUnit]:
        converted = np.array([
            quantity(value, units_value).to("MW").magnitude for value in series
        ], dtype=float)
        return converted

    if units in [EnergyUnit.WH, EnergyUnit.KWH, EnergyUnit.MWH, EnergyUnit.GWH, EnergyUnit.MMBTU] or units_value in [
        it.value for it in EnergyUnit
    ]:
        converted_energy_mwh = np.array([
            quantity(value, units_value).to("MWh").magnitude for value in series
        ], dtype=float)
        return converted_energy_mwh / time_step_hours

    raise ValueError(f"Unsupported dispatch demand units `{units_value}`. Expected hourly energy or thermal power units.")


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

        units = getattr(model.surfaceplant.HeatingDemand, "CurrentYUnits", EnergyUnit.KWH)
        series_mw = _series_to_mw(series, units, time_step_hours=1.0)
        return DemandProfile(series=series_mw, units=PowerUnit.MW.value, num_timesteps=len(series_mw))


@dataclass
class DispatchCommand:
    target_flow_fraction: float
    runtime_fraction: float
    is_shut_in: bool
    target_thermal_demand_mw: float = 0.0


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
    hourly_thermal_demand: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

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
            hourly_thermal_demand=zeros.copy(),
        )


class ReservoirRecoveryModel(ABC):
    @abstractmethod
    def update(self, state: float, dt_hours: float, is_shut_in: bool) -> float:
        raise NotImplementedError


class NoRecoveryModel(ReservoirRecoveryModel):
    def update(self, state: float, dt_hours: float, is_shut_in: bool) -> float:
        return state


class ReducedOrderRecoveryModel(ReservoirRecoveryModel):
    def __init__(
        self,
        equilibrium_state: float,
        recovery_time_constant_hours: float,
        state_bounds: tuple[float, float],
    ):
        self._equilibrium_state = float(equilibrium_state)
        self._recovery_time_constant_hours = max(float(recovery_time_constant_hours), 1.0)
        self._state_min = float(min(state_bounds))
        self._state_max = float(max(state_bounds))

    def update(self, state: float, dt_hours: float, is_shut_in: bool) -> float:
        clipped_state = float(np.clip(state, self._state_min, self._state_max))
        if not is_shut_in or dt_hours <= 0:
            return clipped_state

        recovery_fraction = 1.0 - np.exp(-dt_hours / self._recovery_time_constant_hours)
        recovered_state = clipped_state + (self._equilibrium_state - clipped_state) * recovery_fraction
        return float(np.clip(recovered_state, self._state_min, self._state_max))


class DispatchStrategy(ABC):
    @abstractmethod
    def dispatch(self, timestep_state: dict[str, Any], demand: float) -> DispatchCommand:
        raise NotImplementedError


class DemandFollowingDispatchStrategy(DispatchStrategy):
    def dispatch(self, timestep_state: dict[str, Any], demand: float) -> DispatchCommand:
        if demand <= 0:
            return DispatchCommand(
                target_flow_fraction=0.0,
                runtime_fraction=0.0,
                is_shut_in=True,
                target_thermal_demand_mw=0.0,
            )

        max_flow_fraction = timestep_state.get("maximum_dispatch_flow_fraction", 1.0)
        min_flow_fraction = timestep_state.get("minimum_dispatch_flow_fraction", 0.0)
        min_runtime_fraction = timestep_state.get("minimum_dispatch_runtime_fraction", 0.0)
        nominal_heat_output_mw = timestep_state.get("nominal_heat_output_mw", 0.0)

        if nominal_heat_output_mw <= 0:
            return DispatchCommand(
                target_flow_fraction=0.0,
                runtime_fraction=0.0,
                is_shut_in=True,
                target_thermal_demand_mw=demand,
            )

        required_fraction = demand / nominal_heat_output_mw
        if required_fraction <= 0:
            return DispatchCommand(
                target_flow_fraction=0.0,
                runtime_fraction=0.0,
                is_shut_in=True,
                target_thermal_demand_mw=0.0,
            )

        if min_flow_fraction > 0 and required_fraction < min_flow_fraction:
            runtime_fraction = required_fraction / min_flow_fraction
            runtime_fraction = max(runtime_fraction, min_runtime_fraction)
            runtime_fraction = min(runtime_fraction, 1.0)
            flow_fraction = min_flow_fraction
        else:
            flow_fraction = min(required_fraction, max_flow_fraction)
            runtime_fraction = 1.0

        return DispatchCommand(
            target_flow_fraction=max(0.0, min(flow_fraction, max_flow_fraction)),
            runtime_fraction=max(0.0, min(runtime_fraction, 1.0)),
            is_shut_in=False,
            target_thermal_demand_mw=demand,
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


class CylindricalDispatchPlantAdapter(DispatchPlantAdapter):
    def initialize(self, model: "Model", design_state: dict[str, Any]) -> None:
        self._model = model
        model.reserv.Calculate(model)
        model.wellbores.Calculate(model)

        self._nominal_flow_kg_per_sec = model.wellbores.prodwellflowrate.value
        self._nprod = model.wellbores.nprod.value
        self._tinj = model.wellbores.Tinj.value
        self._cpwater = model.reserv.cpwater.value
        self._enduse_efficiency = model.surfaceplant.enduse_efficiency_factor.value
        self._initial_heat_content_pj = model.reserv.InitialReservoirHeatContent.value
        self._remaining_heat_content_pj = self._initial_heat_content_pj
        self._initial_reservoir_temperature_c = model.reserv.Trock.value
        self._reservoir_temperature_span_c = max(self._initial_reservoir_temperature_c - self._tinj, 0.0)
        if self._initial_heat_content_pj <= 0 or self._reservoir_temperature_span_c <= 0:
            raise ValueError(
                "Cylindrical dispatchable mode requires a positive initial reservoir heat content. "
                "Check cylindrical reservoir geometry and ensure `Number of Multilateral Sections` is greater than zero."
            )
        self._base_temp_drop_c = float(np.atleast_1d(model.wellbores.ProdTempDrop.value)[0])
        self._base_pumping_power_mw = float(np.atleast_1d(model.wellbores.PumpingPower.value)[0])
        self._base_injection_pumping_power_mw = float(np.atleast_1d(model.wellbores.PumpingPowerInj.value)[0])
        self._base_production_pumping_power_mw = float(np.atleast_1d(model.wellbores.PumpingPowerProd.value)[0])
        self._maximum_dispatch_flow_fraction = model.surfaceplant.maximum_dispatch_flow_fraction.value
        self._recovery_model: ReservoirRecoveryModel = ReducedOrderRecoveryModel(
            equilibrium_state=self._initial_heat_content_pj,
            recovery_time_constant_hours=24.0 * 30.0,
            state_bounds=(0.0, self._initial_heat_content_pj),
        )

    def _current_reservoir_temperature(self) -> float:
        if self._initial_heat_content_pj <= 0 or self._reservoir_temperature_span_c <= 0:
            return self._tinj

        fraction_remaining = max(min(self._remaining_heat_content_pj / self._initial_heat_content_pj, 1.0), 0.0)
        return self._tinj + (fraction_remaining * self._reservoir_temperature_span_c)

    def _temperature_drop_for_flow_fraction(self, flow_fraction: float, current_reservoir_temperature_c: float) -> float:
        if flow_fraction <= 0:
            return 0.0

        scaled_temp_drop = self._base_temp_drop_c / max(flow_fraction, 0.25)
        return min(max(scaled_temp_drop, 0.0), max(current_reservoir_temperature_c - self._tinj, 0.0))

    def thermal_state_for_flow_fraction(self, flow_fraction: float) -> dict[str, float]:
        if flow_fraction <= 0:
            current_reservoir_temperature_c = self._current_reservoir_temperature()
            return {
                "produced_temperature_c": current_reservoir_temperature_c,
                "useful_heat_mw": 0.0,
                "extracted_heat_mw": 0.0,
                "pumping_power_mw": 0.0,
                "production_pumping_power_mw": 0.0,
                "injection_pumping_power_mw": 0.0,
                "actual_flow_kg_per_sec": 0.0,
            }

        current_reservoir_temperature_c = self._current_reservoir_temperature()
        actual_flow_kg_per_sec = self._nominal_flow_kg_per_sec * flow_fraction
        temp_drop_c = self._temperature_drop_for_flow_fraction(flow_fraction, current_reservoir_temperature_c)
        produced_temperature_c = max(self._tinj, current_reservoir_temperature_c - temp_drop_c)
        extracted_heat_mw = (
            self._nprod * actual_flow_kg_per_sec * self._cpwater * max(produced_temperature_c - self._tinj, 0.0) / 1.0e6
        )
        useful_heat_mw = extracted_heat_mw * self._enduse_efficiency
        flow_scale = flow_fraction ** 3
        production_pumping_power_mw = self._base_production_pumping_power_mw * flow_scale
        injection_pumping_power_mw = self._base_injection_pumping_power_mw * flow_scale
        pumping_power_mw = self._base_pumping_power_mw * flow_scale

        return {
            "produced_temperature_c": produced_temperature_c,
            "useful_heat_mw": useful_heat_mw,
            "extracted_heat_mw": extracted_heat_mw,
            "pumping_power_mw": pumping_power_mw,
            "production_pumping_power_mw": production_pumping_power_mw,
            "injection_pumping_power_mw": injection_pumping_power_mw,
            "actual_flow_kg_per_sec": actual_flow_kg_per_sec,
        }

    def evaluate_timestep(self, dispatch_command: DispatchCommand, timestep_index: int) -> DispatchTimestepResult:
        current_reservoir_temperature_c = self._current_reservoir_temperature()
        if dispatch_command.is_shut_in or dispatch_command.target_flow_fraction <= 0 or dispatch_command.runtime_fraction <= 0:
            self._remaining_heat_content_pj = self._recovery_model.update(
                self._remaining_heat_content_pj,
                dt_hours=1.0,
                is_shut_in=True,
            )
            return DispatchTimestepResult(
                produced_temperature=self._current_reservoir_temperature(),
                runtime_fraction=0.0,
            )

        thermal_state = self.thermal_state_for_flow_fraction(dispatch_command.target_flow_fraction)
        potential_useful_heat_mw = thermal_state["useful_heat_mw"] * dispatch_command.runtime_fraction
        served_thermal_demand_mw = min(dispatch_command.target_thermal_demand_mw, potential_useful_heat_mw)
        unmet_thermal_demand_mw = max(dispatch_command.target_thermal_demand_mw - served_thermal_demand_mw, 0.0)

        extracted_energy_kwh = thermal_state["extracted_heat_mw"] * dispatch_command.runtime_fraction * 1000.0
        self._remaining_heat_content_pj = max(
            self._remaining_heat_content_pj - (extracted_energy_kwh * 3600.0 * 1000.0 / 1.0e15),
            0.0,
        )
        shut_in_hours = max(1.0 - dispatch_command.runtime_fraction, 0.0)
        self._remaining_heat_content_pj = self._recovery_model.update(
            self._remaining_heat_content_pj,
            dt_hours=shut_in_hours,
            is_shut_in=shut_in_hours > 0.0,
        )

        return DispatchTimestepResult(
            produced_temperature=thermal_state["produced_temperature_c"],
            plant_outlet_thermal_power=potential_useful_heat_mw,
            pumping_power=thermal_state["pumping_power_mw"] * dispatch_command.runtime_fraction,
            served_thermal_demand=served_thermal_demand_mw,
            unmet_thermal_demand=unmet_thermal_demand_mw,
            actual_flow=thermal_state["actual_flow_kg_per_sec"],
            runtime_fraction=dispatch_command.runtime_fraction,
        )

    def design_metrics(self) -> dict[str, float]:
        design_state = self.thermal_state_for_flow_fraction(self._maximum_dispatch_flow_fraction)
        return {
            "design_heat_extracted_mw": design_state["extracted_heat_mw"],
            "design_heat_produced_mw": design_state["useful_heat_mw"],
            "design_pumping_power_mw": design_state["pumping_power_mw"],
            "design_pumping_power_prod_mw": design_state["production_pumping_power_mw"],
            "design_pumping_power_inj_mw": design_state["injection_pumping_power_mw"],
            "design_flow_kg_per_sec": design_state["actual_flow_kg_per_sec"],
        }

    def finalize(self) -> None:
        return None


class AnalyticalReservoirDispatchPlantAdapter(DispatchPlantAdapter):
    def __init__(self, supported_reservoir_name: str):
        self._supported_reservoir_name = supported_reservoir_name

    def initialize(self, model: "Model", design_state: dict[str, Any]) -> None:
        self._model = model
        model.reserv.Calculate(model)
        model.wellbores.Calculate(model)

        self._nominal_flow_kg_per_sec = model.wellbores.prodwellflowrate.value
        self._nprod = model.wellbores.nprod.value
        self._tinj = model.wellbores.Tinj.value
        self._cpwater = model.reserv.cpwater.value
        self._enduse_efficiency = model.surfaceplant.enduse_efficiency_factor.value
        self._maximum_dispatch_flow_fraction = model.surfaceplant.maximum_dispatch_flow_fraction.value

        self._baseline_produced_temperature = _as_baseline_series(
            model.wellbores.ProducedTemperature.value,
            "ProducedTemperature",
        )
        baseline_length = len(self._baseline_produced_temperature)
        self._baseline_pumping_power_mw = _coerce_series_length(
            model.wellbores.PumpingPower.value,
            baseline_length,
            "PumpingPower",
        )
        self._baseline_pumping_power_prod_mw = _coerce_series_length(
            model.wellbores.PumpingPowerProd.value,
            baseline_length,
            "PumpingPowerProd",
        )
        self._baseline_pumping_power_inj_mw = _coerce_series_length(
            model.wellbores.PumpingPowerInj.value,
            baseline_length,
            "PumpingPowerInj",
        )
        self._baseline_heat_extracted_mw = (
            self._nprod
            * self._nominal_flow_kg_per_sec
            * self._cpwater
            * np.maximum(self._baseline_produced_temperature - self._tinj, 0.0)
            / 1.0e6
        )
        self._baseline_cumulative_extracted_kwh = np.cumsum(self._baseline_heat_extracted_mw * 1000.0)
        self._baseline_total_extracted_kwh = float(self._baseline_cumulative_extracted_kwh[-1]) if self._baseline_cumulative_extracted_kwh.size > 0 else 0.0
        self._cumulative_extracted_kwh = 0.0

        if self._baseline_total_extracted_kwh <= 0:
            raise ValueError(
                f"{self._supported_reservoir_name} dispatchable mode requires a positive baseline extracted-energy profile."
            )
        self._recovery_model: ReservoirRecoveryModel = ReducedOrderRecoveryModel(
            equilibrium_state=0.0,
            recovery_time_constant_hours=24.0 * 21.0,
            state_bounds=(0.0, self._baseline_total_extracted_kwh),
        )

    def _baseline_index_for_depletion(self) -> int:
        if self._baseline_total_extracted_kwh <= 0:
            return 0

        depletion_fraction = min(max(self._cumulative_extracted_kwh / self._baseline_total_extracted_kwh, 0.0), 1.0)
        return min(int(round(depletion_fraction * (len(self._baseline_produced_temperature) - 1))), len(self._baseline_produced_temperature) - 1)

    def thermal_state_for_flow_fraction(self, flow_fraction: float) -> dict[str, float]:
        baseline_index = self._baseline_index_for_depletion()
        baseline_temperature_c = self._baseline_produced_temperature[baseline_index]
        baseline_heat_extracted_mw = self._baseline_heat_extracted_mw[baseline_index]
        baseline_pumping_power_mw = self._baseline_pumping_power_mw[baseline_index]
        baseline_pumping_power_prod_mw = self._baseline_pumping_power_prod_mw[baseline_index]
        baseline_pumping_power_inj_mw = self._baseline_pumping_power_inj_mw[baseline_index]

        if flow_fraction <= 0:
            return {
                "produced_temperature_c": baseline_temperature_c,
                "useful_heat_mw": 0.0,
                "extracted_heat_mw": 0.0,
                "pumping_power_mw": 0.0,
                "production_pumping_power_mw": 0.0,
                "injection_pumping_power_mw": 0.0,
                "actual_flow_kg_per_sec": 0.0,
            }

        actual_flow_kg_per_sec = self._nominal_flow_kg_per_sec * flow_fraction
        extracted_heat_mw = baseline_heat_extracted_mw * flow_fraction
        useful_heat_mw = extracted_heat_mw * self._enduse_efficiency
        flow_scale = flow_fraction ** 3

        return {
            "produced_temperature_c": baseline_temperature_c,
            "useful_heat_mw": useful_heat_mw,
            "extracted_heat_mw": extracted_heat_mw,
            "pumping_power_mw": baseline_pumping_power_mw * flow_scale,
            "production_pumping_power_mw": baseline_pumping_power_prod_mw * flow_scale,
            "injection_pumping_power_mw": baseline_pumping_power_inj_mw * flow_scale,
            "actual_flow_kg_per_sec": actual_flow_kg_per_sec,
        }

    def evaluate_timestep(self, dispatch_command: DispatchCommand, timestep_index: int) -> DispatchTimestepResult:
        baseline_index = self._baseline_index_for_depletion()
        baseline_temperature_c = self._baseline_produced_temperature[baseline_index]

        if dispatch_command.is_shut_in or dispatch_command.target_flow_fraction <= 0 or dispatch_command.runtime_fraction <= 0:
            self._cumulative_extracted_kwh = self._recovery_model.update(
                self._cumulative_extracted_kwh,
                dt_hours=1.0,
                is_shut_in=True,
            )
            return DispatchTimestepResult(
                produced_temperature=self._baseline_produced_temperature[self._baseline_index_for_depletion()],
                runtime_fraction=0.0,
            )

        thermal_state = self.thermal_state_for_flow_fraction(dispatch_command.target_flow_fraction)
        potential_useful_heat_mw = thermal_state["useful_heat_mw"] * dispatch_command.runtime_fraction
        served_thermal_demand_mw = min(dispatch_command.target_thermal_demand_mw, potential_useful_heat_mw)
        unmet_thermal_demand_mw = max(dispatch_command.target_thermal_demand_mw - served_thermal_demand_mw, 0.0)
        extracted_energy_kwh = thermal_state["extracted_heat_mw"] * dispatch_command.runtime_fraction * 1000.0
        self._cumulative_extracted_kwh += extracted_energy_kwh
        shut_in_hours = max(1.0 - dispatch_command.runtime_fraction, 0.0)
        self._cumulative_extracted_kwh = self._recovery_model.update(
            self._cumulative_extracted_kwh,
            dt_hours=shut_in_hours,
            is_shut_in=shut_in_hours > 0.0,
        )

        return DispatchTimestepResult(
            produced_temperature=thermal_state["produced_temperature_c"],
            plant_outlet_thermal_power=potential_useful_heat_mw,
            pumping_power=thermal_state["pumping_power_mw"] * dispatch_command.runtime_fraction,
            served_thermal_demand=served_thermal_demand_mw,
            unmet_thermal_demand=unmet_thermal_demand_mw,
            actual_flow=thermal_state["actual_flow_kg_per_sec"],
            runtime_fraction=dispatch_command.runtime_fraction,
        )

    def design_metrics(self) -> dict[str, float]:
        design_state = self.thermal_state_for_flow_fraction(self._maximum_dispatch_flow_fraction)
        return {
            "design_heat_extracted_mw": design_state["extracted_heat_mw"],
            "design_heat_produced_mw": design_state["useful_heat_mw"],
            "design_pumping_power_mw": design_state["pumping_power_mw"],
            "design_pumping_power_prod_mw": design_state["production_pumping_power_mw"],
            "design_pumping_power_inj_mw": design_state["injection_pumping_power_mw"],
            "design_flow_kg_per_sec": design_state["actual_flow_kg_per_sec"],
        }

    def finalize(self) -> None:
        return None


class SBTDispatchPlantAdapter(AnalyticalReservoirDispatchPlantAdapter):
    def __init__(self):
        super().__init__("SBTReservoir")

    def initialize(self, model: "Model", design_state: dict[str, Any]) -> None:
        configuration = model.wellbores.Configuration.value
        if not isinstance(configuration, Configuration):
            configuration = Configuration.from_input_string(configuration)

        if configuration == Configuration.COAXIAL:
            raise NotImplementedError(
                "Dispatchable mode for SBTReservoir currently supports only U-loop style configurations; "
                "coaxial SBT is not implemented."
            )

        if configuration not in [Configuration.ULOOP, Configuration.EAVORLOOP]:
            raise ValueError(
                f"Unsupported SBT configuration for dispatchable mode: {_enum_value_or_str(configuration)}"
            )

        super().initialize(model, design_state)


class DispatchAdapterFactory:
    _registry = {
        "CylindricalReservoir": lambda: CylindricalDispatchPlantAdapter(),
        "MPFReservoir": lambda: AnalyticalReservoirDispatchPlantAdapter("MPFReservoir"),
        "LHSReservoir": lambda: AnalyticalReservoirDispatchPlantAdapter("LHSReservoir"),
        "SFReservoir": lambda: AnalyticalReservoirDispatchPlantAdapter("SFReservoir"),
        "UPPReservoir": lambda: AnalyticalReservoirDispatchPlantAdapter("UPPReservoir"),
        "SBTReservoir": lambda: SBTDispatchPlantAdapter(),
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
        dispatch_demand_mw = np.tile(demand_profile.series, model.surfaceplant.plant_lifetime.value)
        model.dispatch_results = DispatchResults.initialize(len(dispatch_demand_mw))
        model.dispatch_adapter = DispatchAdapterFactory.create(model)
        model.dispatch_adapter.initialize(model, design_state={})
        design_metrics = getattr(model.dispatch_adapter, "design_metrics", lambda: {})()

        if not hasattr(model.dispatch_adapter, "thermal_state_for_flow_fraction"):
            model.dispatch_adapter.evaluate_timestep(
                DispatchCommand(target_flow_fraction=0.0, runtime_fraction=0.0, is_shut_in=True),
                0,
            )

        for timestep_index, timestep_demand_mw in enumerate(dispatch_demand_mw):
            nominal_state = model.dispatch_adapter.thermal_state_for_flow_fraction(1.0)
            timestep_state = {
                "nominal_heat_output_mw": nominal_state["useful_heat_mw"],
                "maximum_dispatch_flow_fraction": model.surfaceplant.maximum_dispatch_flow_fraction.value,
                "minimum_dispatch_flow_fraction": model.surfaceplant.minimum_dispatch_flow_fraction.value,
                "minimum_dispatch_runtime_fraction": model.surfaceplant.minimum_dispatch_runtime_fraction.value,
            }
            dispatch_command = self._dispatch_strategy.dispatch(timestep_state, timestep_demand_mw)
            timestep_result = model.dispatch_adapter.evaluate_timestep(dispatch_command, timestep_index)

            model.dispatch_results.hourly_thermal_demand[timestep_index] = timestep_demand_mw
            model.dispatch_results.hourly_produced_temperature[timestep_index] = timestep_result.produced_temperature
            model.dispatch_results.hourly_flow[timestep_index] = timestep_result.actual_flow
            model.dispatch_results.hourly_runtime_fraction[timestep_index] = timestep_result.runtime_fraction
            model.dispatch_results.hourly_demand_served[timestep_index] = timestep_result.served_thermal_demand * 1000.0
            model.dispatch_results.hourly_unmet_demand[timestep_index] = timestep_result.unmet_thermal_demand * 1000.0
            model.dispatch_results.hourly_pumping_power[timestep_index] = timestep_result.pumping_power
            model.dispatch_results.hourly_geothermal_thermal_output[timestep_index] = timestep_result.plant_outlet_thermal_power

        model.dispatch_adapter.finalize()
        model.dispatch_results.summary_metrics.update(design_metrics)
        self._finalize_dispatch_results(model, dispatch_demand_mw, demand_profile.time_step_hours)
        model.economics.Calculate(model)

    @staticmethod
    def _finalize_dispatch_results(model: "Model", dispatch_demand_mw: np.ndarray, time_step_hours: float) -> None:
        timesteps_per_year = 8760
        plant_lifetime_years = model.surfaceplant.plant_lifetime.value
        efficiency = model.surfaceplant.enduse_efficiency_factor.value

        served_heat_kwh = model.dispatch_results.hourly_demand_served
        unmet_heat_kwh = model.dispatch_results.hourly_unmet_demand
        useful_heat_mw = served_heat_kwh / 1000.0
        extracted_heat_mw = useful_heat_mw / efficiency if efficiency > 0 else np.zeros_like(useful_heat_mw)
        pumping_power_mw = model.dispatch_results.hourly_pumping_power

        model.economics.timestepsperyear.value = timesteps_per_year
        model.surfaceplant.utilization_factor.value = float(np.average(model.dispatch_results.hourly_runtime_fraction))

        model.wellbores.ProducedTemperature.value = model.dispatch_results.hourly_produced_temperature.copy()
        model.wellbores.PumpingPower.value = pumping_power_mw.copy()
        model.wellbores.PumpingPowerInj.value = pumping_power_mw.copy()
        model.wellbores.PumpingPowerProd.value = np.zeros_like(pumping_power_mw)
        model.wellbores.redrill.value = 0

        model.surfaceplant.HeatProduced.value = useful_heat_mw.copy()
        model.surfaceplant.HeatExtracted.value = extracted_heat_mw.copy()
        model.surfaceplant.PumpingkWh.value = np.zeros(plant_lifetime_years)
        model.surfaceplant.HeatkWhExtracted.value = np.zeros(plant_lifetime_years)
        model.surfaceplant.HeatkWhProduced.value = np.zeros(plant_lifetime_years)
        model.surfaceplant.RemainingReservoirHeatContent.value = np.zeros(plant_lifetime_years)

        for year_index in range(plant_lifetime_years):
            start = year_index * timesteps_per_year
            end = start + timesteps_per_year
            model.surfaceplant.HeatkWhProduced.value[year_index] = float(np.sum(served_heat_kwh[start:end]))
            model.surfaceplant.HeatkWhExtracted.value[year_index] = float(np.sum(extracted_heat_mw[start:end] * 1000.0))
            model.surfaceplant.PumpingkWh.value[year_index] = float(np.sum(pumping_power_mw[start:end] * 1000.0))

        model.surfaceplant.RemainingReservoirHeatContent.value = model.surfaceplant.remaining_reservoir_heat_content(
            model.reserv.InitialReservoirHeatContent.value,
            model.surfaceplant.HeatkWhExtracted.value,
        )

        annual_served_kwh = model.surfaceplant.HeatkWhProduced.value
        annual_unmet_kwh = np.array([
            float(np.sum(unmet_heat_kwh[year_index * timesteps_per_year:(year_index + 1) * timesteps_per_year]))
            for year_index in range(plant_lifetime_years)
        ])
        dispatch_demand_kwh = dispatch_demand_mw * 1000.0 * time_step_hours
        annual_demand_kwh = np.array([
            float(np.sum(dispatch_demand_kwh[year_index * timesteps_per_year:(year_index + 1) * timesteps_per_year]))
            for year_index in range(plant_lifetime_years)
        ])

        design_heat_produced_mw = model.dispatch_results.summary_metrics.get("design_heat_produced_mw", 0.0)
        average_runtime_fraction = float(np.average(model.dispatch_results.hourly_runtime_fraction))
        average_capacity_factor = float(
            np.average(
                served_heat_kwh / (design_heat_produced_mw * 1000.0)
            )
        ) if design_heat_produced_mw > 0 else 0.0

        model.dispatch_results.annual_aggregates = {
            "annual_served_heat_kwh": annual_served_kwh.tolist(),
            "annual_unmet_heat_kwh": annual_unmet_kwh.tolist(),
            "annual_heat_demand_kwh": annual_demand_kwh.tolist(),
        }
        model.dispatch_results.summary_metrics.update(
            {
                "peak_hourly_demand_mw": float(np.max(dispatch_demand_mw)),
                "peak_served_heat_kwh": float(np.max(served_heat_kwh)),
                "peak_unmet_heat_kwh": float(np.max(unmet_heat_kwh)),
                "annual_served_heat_kwh": float(np.sum(annual_served_kwh)),
                "annual_unmet_heat_kwh": float(np.sum(annual_unmet_kwh)),
                "average_runtime_fraction": average_runtime_fraction,
                "dispatch_capacity_factor": average_capacity_factor,
                "observed_peak_flow_kg_per_sec": float(np.max(model.dispatch_results.hourly_flow)),
            }
        )


def create_operating_mode_strategy(operating_mode: OperatingMode) -> OperatingModeStrategy:
    if operating_mode == OperatingMode.DISPATCHABLE:
        return DispatchableOperatingModeStrategy()

    return BaseloadOperatingModeStrategy()
