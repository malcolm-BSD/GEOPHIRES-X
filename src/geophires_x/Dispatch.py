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
    demand_type: str = "thermal"


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
        if source == DispatchDemandSource.ANNUAL_HEAT_DEMAND:
            parameter = model.surfaceplant.HeatingDemand
            parameter_name = "Annual Heat Demand"
            demand_type = "thermal"
        elif source == DispatchDemandSource.ANNUAL_ELECTRICITY_DEMAND:
            parameter = model.surfaceplant.ElectricityDemand
            parameter_name = "Annual Electricity Demand"
            demand_type = "electric"
        else:
            raise ValueError(f"Unsupported dispatch demand source: {_enum_value_or_str(source)}")

        raw_series = parameter.value
        if len(raw_series) == 0:
            raise ValueError(
                f"Dispatchable mode requires `{parameter_name}` to be provided as an hourly demand profile."
            )

        series = _as_float_series(raw_series)
        if len(series) != 8760:
            raise ValueError(
                f"Dispatchable mode requires an hourly one-year demand profile with 8760 timesteps; received {len(series)}."
            )

        units = getattr(parameter, "CurrentYUnits", EnergyUnit.KWH)
        series_mw = _series_to_mw(series, units, time_step_hours=1.0)
        return DemandProfile(
            series=series_mw,
            units=PowerUnit.MW.value,
            num_timesteps=len(series_mw),
            demand_type=demand_type,
        )


def _dispatch_surfaceplant_mode(model: "Model") -> str:
    enduse_option = model.surfaceplant.enduse_option.value
    if enduse_option == EndUseOptions.HEAT:
        return "thermal"
    if enduse_option == EndUseOptions.ELECTRICITY:
        if model.surfaceplant.plant_type.value not in [PlantType.SUB_CRITICAL_ORC, PlantType.SUPER_CRITICAL_ORC]:
            raise ValueError(
                "Dispatchable electricity mode currently supports only subcritical and supercritical ORC plants."
            )
        return "electric"

    raise ValueError("Dispatchable mode currently supports only direct-use heat and pure electricity cases.")


def _get_orc_coefficients(model: "Model") -> tuple[float, float, float, float, float, float, float, float, float, float, float, float]:
    ambient_temperature = model.surfaceplant.ambient_temperature.value
    plant_type = model.surfaceplant.plant_type.value

    if plant_type == PlantType.SUB_CRITICAL_ORC:
        if ambient_temperature < 15.0:
            return (-8.3806e-2, 2.746e-3, 0.0, -9.1841e-2, 2.713e-3, 0.0, 55.6, 0.0894, 0.0, 62.6, 0.0894, 0.0)
        return (-9.1841e-2, 2.713e-3, 0.0, -1.012e-1, 2.676e-3, 0.0, 62.6, 0.0894, 0.0, 69.6, 0.0894, 0.0)

    if plant_type == PlantType.SUPER_CRITICAL_ORC:
        if ambient_temperature < 15.0:
            return (-3.78e-1, 7.604e-3, -1.55e-5, -3.7915e-1, 7.4268e-3, -1.499e-5, 49.26, 0.02, 0.0, 56.26, 0.02, 0.0)
        return (-3.7915e-1, 7.4268e-3, -1.499e-5, -4.041e-1, 7.55136e-3, -1.55e-5, 56.26, 0.02, 0.0, 63.26, 0.02, 0.0)

    raise ValueError("Dispatchable electricity mode currently supports only subcritical and supercritical ORC plants.")


def _electricity_dispatch_state(
    model: "Model",
    produced_temperature_c: float,
    actual_flow_kg_per_sec: float,
    pumping_power_mw: float,
) -> dict[str, float]:
    surfaceplant = model.surfaceplant
    produced_temperature = np.array([produced_temperature_c], dtype=float)
    plant_entering_temperature = surfaceplant.power_plant_entering_temperature(
        surfaceplant.enduse_option.value,
        np.array([0.0], dtype=float),
        surfaceplant.T_chp_bottom.value,
        produced_temperature,
    )
    availability = surfaceplant.availability_water(
        surfaceplant.ambient_temperature.value,
        plant_entering_temperature,
        surfaceplant.ambient_temperature.value,
    )
    adjusted_tinj, reinjection_temperature, etau = surfaceplant.reinjection_temperature(
        model,
        surfaceplant.ambient_temperature.value,
        plant_entering_temperature,
        model.wellbores.Tinj.value,
        *_get_orc_coefficients(model),
    )
    electricity_produced, heat_extracted, heat_produced, heat_towards_electricity = surfaceplant.electricity_heat_production(
        surfaceplant.enduse_option.value,
        availability,
        etau,
        model.wellbores.nprod.value,
        actual_flow_kg_per_sec,
        model.reserv.cpwater.value,
        produced_temperature,
        adjusted_tinj,
        reinjection_temperature,
        surfaceplant.T_chp_bottom.value,
        surfaceplant.enduse_efficiency_factor.value,
        surfaceplant.chp_fraction.value,
    )

    gross_electricity_mw = float(np.atleast_1d(electricity_produced)[0])
    heat_extracted_mw = float(np.atleast_1d(heat_extracted)[0])
    heat_towards_electricity_mw = float(np.atleast_1d(heat_towards_electricity)[0])
    net_electricity_mw = gross_electricity_mw - pumping_power_mw

    return {
        "dispatch_output_mw": max(net_electricity_mw, 0.0),
        "gross_electricity_mw": gross_electricity_mw,
        "net_electricity_mw": net_electricity_mw,
        "heat_extracted_mw": heat_extracted_mw,
        "heat_produced_mw": float(np.atleast_1d(heat_produced)[0]) if np.size(heat_produced) > 0 else 0.0,
        "plant_entering_temperature_c": float(np.atleast_1d(plant_entering_temperature)[0]),
        "availability_mj_per_kg": float(np.atleast_1d(availability)[0]),
        "reinjection_temperature_c": float(np.atleast_1d(reinjection_temperature)[0]),
        "first_law_efficiency": (
            net_electricity_mw / heat_towards_electricity_mw if heat_towards_electricity_mw > 0 else 0.0
        ),
    }


@dataclass
class DispatchCommand:
    target_flow_fraction: float
    runtime_fraction: float
    is_shut_in: bool
    target_demand_mw: float = 0.0


@dataclass
class DispatchTimestepResult:
    produced_temperature: float = 0.0
    plant_outlet_thermal_power: float = 0.0
    plant_outlet_electric_power: float = 0.0
    gross_electric_power: float = 0.0
    extracted_heat_power: float = 0.0
    pumping_power: float = 0.0
    served_demand: float = 0.0
    unmet_demand: float = 0.0
    actual_flow: float = 0.0
    runtime_fraction: float = 0.0
    plant_entering_temperature: float = 0.0
    reinjection_temperature: float = 0.0
    availability: float = 0.0
    first_law_efficiency: float = 0.0


@dataclass
class DispatchResults:
    hourly_produced_temperature: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hourly_flow: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hourly_runtime_fraction: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hourly_demand_served: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hourly_unmet_demand: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hourly_pumping_power: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hourly_geothermal_thermal_output: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hourly_geothermal_electric_output: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hourly_gross_electric_output: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hourly_heat_extracted: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hourly_tentering_powerplant: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hourly_reinjection_temperature: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hourly_availability: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hourly_first_law_efficiency: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    annual_aggregates: dict[str, float] = field(default_factory=dict)
    summary_metrics: dict[str, float] = field(default_factory=dict)
    hourly_thermal_demand: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    analysis_start_year: int = 1
    analysis_end_year: int = 2
    simulation_start_hour: int = 1
    demand_type: str = "thermal"

    @classmethod
    def initialize(
        cls,
        num_timesteps: int,
        analysis_start_year: int = 1,
        analysis_end_year: int = 2,
        simulation_start_hour: int = 1,
        demand_type: str = "thermal",
    ) -> "DispatchResults":
        zeros = np.zeros(num_timesteps, dtype=float)
        return cls(
            hourly_produced_temperature=zeros.copy(),
            hourly_flow=zeros.copy(),
            hourly_runtime_fraction=zeros.copy(),
            hourly_demand_served=zeros.copy(),
            hourly_unmet_demand=zeros.copy(),
            hourly_pumping_power=zeros.copy(),
            hourly_geothermal_thermal_output=zeros.copy(),
            hourly_geothermal_electric_output=zeros.copy(),
            hourly_gross_electric_output=zeros.copy(),
            hourly_heat_extracted=zeros.copy(),
            hourly_tentering_powerplant=zeros.copy(),
            hourly_reinjection_temperature=zeros.copy(),
            hourly_availability=zeros.copy(),
            hourly_first_law_efficiency=zeros.copy(),
            hourly_thermal_demand=zeros.copy(),
            analysis_start_year=analysis_start_year,
            analysis_end_year=analysis_end_year,
            simulation_start_hour=simulation_start_hour,
            demand_type=demand_type,
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
                target_demand_mw=0.0,
            )

        max_flow_fraction = timestep_state.get("maximum_dispatch_flow_fraction", 1.0)
        min_flow_fraction = timestep_state.get("minimum_dispatch_flow_fraction", 0.0)
        min_runtime_fraction = timestep_state.get("minimum_dispatch_runtime_fraction", 0.0)
        nominal_output_mw = timestep_state.get("nominal_output_mw", 0.0)

        if nominal_output_mw <= 0:
            return DispatchCommand(
                target_flow_fraction=0.0,
                runtime_fraction=0.0,
                is_shut_in=True,
                target_demand_mw=demand,
            )

        required_fraction = demand / nominal_output_mw
        if required_fraction <= 0:
            return DispatchCommand(
                target_flow_fraction=0.0,
                runtime_fraction=0.0,
                is_shut_in=True,
                target_demand_mw=0.0,
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
            target_demand_mw=demand,
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
        self._dispatch_mode = _dispatch_surfaceplant_mode(model)
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
            state = {
                "produced_temperature_c": current_reservoir_temperature_c,
                "useful_heat_mw": 0.0,
                "extracted_heat_mw": 0.0,
                "pumping_power_mw": 0.0,
                "production_pumping_power_mw": 0.0,
                "injection_pumping_power_mw": 0.0,
                "actual_flow_kg_per_sec": 0.0,
            }
            if self._dispatch_mode == "thermal":
                state["dispatch_output_mw"] = 0.0
            else:
                state.update(_electricity_dispatch_state(self._model, current_reservoir_temperature_c, 0.0, 0.0))
            return state

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

        state = {
            "produced_temperature_c": produced_temperature_c,
            "useful_heat_mw": useful_heat_mw,
            "extracted_heat_mw": extracted_heat_mw,
            "pumping_power_mw": pumping_power_mw,
            "production_pumping_power_mw": production_pumping_power_mw,
            "injection_pumping_power_mw": injection_pumping_power_mw,
            "actual_flow_kg_per_sec": actual_flow_kg_per_sec,
        }
        if self._dispatch_mode == "thermal":
            state["dispatch_output_mw"] = useful_heat_mw
        else:
            state.update(
                _electricity_dispatch_state(self._model, produced_temperature_c, actual_flow_kg_per_sec, pumping_power_mw)
            )
        return state

    def evaluate_timestep(self, dispatch_command: DispatchCommand, timestep_index: int) -> DispatchTimestepResult:
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
        potential_output_mw = thermal_state["dispatch_output_mw"] * dispatch_command.runtime_fraction
        served_demand_mw = min(dispatch_command.target_demand_mw, potential_output_mw)
        unmet_demand_mw = max(dispatch_command.target_demand_mw - served_demand_mw, 0.0)

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
            plant_outlet_thermal_power=thermal_state["useful_heat_mw"] * dispatch_command.runtime_fraction,
            plant_outlet_electric_power=thermal_state.get("net_electricity_mw", 0.0) * dispatch_command.runtime_fraction,
            gross_electric_power=thermal_state.get("gross_electricity_mw", 0.0) * dispatch_command.runtime_fraction,
            extracted_heat_power=thermal_state["extracted_heat_mw"] * dispatch_command.runtime_fraction,
            pumping_power=thermal_state["pumping_power_mw"] * dispatch_command.runtime_fraction,
            served_demand=served_demand_mw,
            unmet_demand=unmet_demand_mw,
            actual_flow=thermal_state["actual_flow_kg_per_sec"],
            runtime_fraction=dispatch_command.runtime_fraction,
            plant_entering_temperature=thermal_state.get("plant_entering_temperature_c", 0.0),
            reinjection_temperature=thermal_state.get("reinjection_temperature_c", 0.0),
            availability=thermal_state.get("availability_mj_per_kg", 0.0),
            first_law_efficiency=thermal_state.get("first_law_efficiency", 0.0),
        )

    def design_metrics(self) -> dict[str, float]:
        design_state = self.thermal_state_for_flow_fraction(self._maximum_dispatch_flow_fraction)
        metrics = {
            "design_heat_extracted_mw": design_state["extracted_heat_mw"],
            "design_heat_produced_mw": design_state["useful_heat_mw"],
            "design_pumping_power_mw": design_state["pumping_power_mw"],
            "design_pumping_power_prod_mw": design_state["production_pumping_power_mw"],
            "design_pumping_power_inj_mw": design_state["injection_pumping_power_mw"],
            "design_flow_kg_per_sec": self._nominal_flow_kg_per_sec,
        }
        if self._dispatch_mode == "electric":
            metrics["design_gross_electricity_produced_mw"] = design_state.get("gross_electricity_mw", 0.0)
            metrics["design_net_electricity_produced_mw"] = max(design_state.get("net_electricity_mw", 0.0), 0.0)
        return metrics

    def finalize(self) -> None:
        return None


class AnalyticalReservoirDispatchPlantAdapter(DispatchPlantAdapter):
    def __init__(self, supported_reservoir_name: str):
        self._supported_reservoir_name = supported_reservoir_name

    def initialize(self, model: "Model", design_state: dict[str, Any]) -> None:
        self._model = model
        self._dispatch_mode = _dispatch_surfaceplant_mode(model)
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
            state = {
                "produced_temperature_c": baseline_temperature_c,
                "useful_heat_mw": 0.0,
                "extracted_heat_mw": 0.0,
                "pumping_power_mw": 0.0,
                "production_pumping_power_mw": 0.0,
                "injection_pumping_power_mw": 0.0,
                "actual_flow_kg_per_sec": 0.0,
            }
            if self._dispatch_mode == "thermal":
                state["dispatch_output_mw"] = 0.0
            else:
                state.update(_electricity_dispatch_state(self._model, baseline_temperature_c, 0.0, 0.0))
            return state

        actual_flow_kg_per_sec = self._nominal_flow_kg_per_sec * flow_fraction
        extracted_heat_mw = baseline_heat_extracted_mw * flow_fraction
        useful_heat_mw = extracted_heat_mw * self._enduse_efficiency
        flow_scale = flow_fraction ** 3

        state = {
            "produced_temperature_c": baseline_temperature_c,
            "useful_heat_mw": useful_heat_mw,
            "extracted_heat_mw": extracted_heat_mw,
            "pumping_power_mw": baseline_pumping_power_mw * flow_scale,
            "production_pumping_power_mw": baseline_pumping_power_prod_mw * flow_scale,
            "injection_pumping_power_mw": baseline_pumping_power_inj_mw * flow_scale,
            "actual_flow_kg_per_sec": actual_flow_kg_per_sec,
        }
        if self._dispatch_mode == "thermal":
            state["dispatch_output_mw"] = useful_heat_mw
        else:
            state.update(
                _electricity_dispatch_state(
                    self._model,
                    baseline_temperature_c,
                    actual_flow_kg_per_sec,
                    baseline_pumping_power_mw * flow_scale,
                )
            )
        return state

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
        potential_output_mw = thermal_state["dispatch_output_mw"] * dispatch_command.runtime_fraction
        served_demand_mw = min(dispatch_command.target_demand_mw, potential_output_mw)
        unmet_demand_mw = max(dispatch_command.target_demand_mw - served_demand_mw, 0.0)
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
            plant_outlet_thermal_power=thermal_state["useful_heat_mw"] * dispatch_command.runtime_fraction,
            plant_outlet_electric_power=thermal_state.get("net_electricity_mw", 0.0) * dispatch_command.runtime_fraction,
            gross_electric_power=thermal_state.get("gross_electricity_mw", 0.0) * dispatch_command.runtime_fraction,
            extracted_heat_power=thermal_state["extracted_heat_mw"] * dispatch_command.runtime_fraction,
            pumping_power=thermal_state["pumping_power_mw"] * dispatch_command.runtime_fraction,
            served_demand=served_demand_mw,
            unmet_demand=unmet_demand_mw,
            actual_flow=thermal_state["actual_flow_kg_per_sec"],
            runtime_fraction=dispatch_command.runtime_fraction,
            plant_entering_temperature=thermal_state.get("plant_entering_temperature_c", 0.0),
            reinjection_temperature=thermal_state.get("reinjection_temperature_c", 0.0),
            availability=thermal_state.get("availability_mj_per_kg", 0.0),
            first_law_efficiency=thermal_state.get("first_law_efficiency", 0.0),
        )

    def design_metrics(self) -> dict[str, float]:
        design_state = self.thermal_state_for_flow_fraction(self._maximum_dispatch_flow_fraction)
        metrics = {
            "design_heat_extracted_mw": design_state["extracted_heat_mw"],
            "design_heat_produced_mw": design_state["useful_heat_mw"],
            "design_pumping_power_mw": design_state["pumping_power_mw"],
            "design_pumping_power_prod_mw": design_state["production_pumping_power_mw"],
            "design_pumping_power_inj_mw": design_state["injection_pumping_power_mw"],
            "design_flow_kg_per_sec": self._nominal_flow_kg_per_sec,
        }
        if self._dispatch_mode == "electric":
            metrics["design_gross_electricity_produced_mw"] = design_state.get("gross_electricity_mw", 0.0)
            metrics["design_net_electricity_produced_mw"] = max(design_state.get("net_electricity_mw", 0.0), 0.0)
        return metrics

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
        flow_strategy = model.surfaceplant.dispatch_flow_strategy.value
        if not isinstance(flow_strategy, DispatchFlowStrategy):
            flow_strategy = DispatchFlowStrategy.from_input_string(flow_strategy)

        if flow_strategy != DispatchFlowStrategy.DEMAND_FOLLOWING:
            raise ValueError(
                f"Unsupported dispatch flow strategy: {_enum_value_or_str(flow_strategy)}"
            )

        demand_profile = DemandProfileFactory.from_model(model)
        dispatch_mode = _dispatch_surfaceplant_mode(model)
        plant_lifetime_years = model.surfaceplant.plant_lifetime.value
        analysis_start_year = model.surfaceplant.dispatch_analysis_start_year.value
        analysis_end_year = model.surfaceplant.dispatch_analysis_end_year.value
        dispatch_demand_mw = np.tile(demand_profile.series, plant_lifetime_years)[
            (analysis_start_year - 1) * 8760:(analysis_end_year - 1) * 8760
        ]
        model.dispatch_results = DispatchResults.initialize(
            len(dispatch_demand_mw),
            analysis_start_year=analysis_start_year,
            analysis_end_year=analysis_end_year,
            simulation_start_hour=((analysis_start_year - 1) * 8760) + 1,
            demand_type=demand_profile.demand_type,
        )
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
                "nominal_output_mw": nominal_state["dispatch_output_mw"],
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
            model.dispatch_results.hourly_demand_served[timestep_index] = timestep_result.served_demand * 1000.0
            model.dispatch_results.hourly_unmet_demand[timestep_index] = timestep_result.unmet_demand * 1000.0
            model.dispatch_results.hourly_pumping_power[timestep_index] = timestep_result.pumping_power
            model.dispatch_results.hourly_geothermal_thermal_output[timestep_index] = timestep_result.plant_outlet_thermal_power
            model.dispatch_results.hourly_geothermal_electric_output[timestep_index] = timestep_result.plant_outlet_electric_power
            model.dispatch_results.hourly_gross_electric_output[timestep_index] = timestep_result.gross_electric_power
            model.dispatch_results.hourly_heat_extracted[timestep_index] = timestep_result.extracted_heat_power
            model.dispatch_results.hourly_tentering_powerplant[timestep_index] = timestep_result.plant_entering_temperature
            model.dispatch_results.hourly_reinjection_temperature[timestep_index] = timestep_result.reinjection_temperature
            model.dispatch_results.hourly_availability[timestep_index] = timestep_result.availability
            model.dispatch_results.hourly_first_law_efficiency[timestep_index] = timestep_result.first_law_efficiency

        model.dispatch_adapter.finalize()
        model.dispatch_results.summary_metrics.update(design_metrics)
        self._finalize_dispatch_results(
            model,
            dispatch_demand_mw,
            demand_profile.time_step_hours,
            demand_type=dispatch_mode,
            analysis_start_year=analysis_start_year,
            analysis_end_year=analysis_end_year,
        )
        model.economics.Calculate(model)

    @staticmethod
    def _finalize_dispatch_results(
        model: "Model",
        dispatch_demand_mw: np.ndarray,
        time_step_hours: float,
        demand_type: str,
        analysis_start_year: int,
        analysis_end_year: int,
    ) -> None:
        timesteps_per_year = 8760
        plant_lifetime_years = model.surfaceplant.plant_lifetime.value
        efficiency = model.surfaceplant.enduse_efficiency_factor.value
        total_timesteps = plant_lifetime_years * timesteps_per_year
        analysis_start_index = (analysis_start_year - 1) * timesteps_per_year
        analysis_end_index = analysis_start_index + len(model.dispatch_results.hourly_produced_temperature)

        def _full_timeline(values: np.ndarray) -> np.ndarray:
            full_values = np.zeros(total_timesteps, dtype=float)
            full_values[analysis_start_index:analysis_end_index] = values.copy()
            return full_values

        full_hourly_produced_temperature = _full_timeline(model.dispatch_results.hourly_produced_temperature)
        full_hourly_flow = _full_timeline(model.dispatch_results.hourly_flow)
        full_hourly_runtime_fraction = _full_timeline(model.dispatch_results.hourly_runtime_fraction)
        full_served_demand_kwh = _full_timeline(model.dispatch_results.hourly_demand_served)
        full_unmet_demand_kwh = _full_timeline(model.dispatch_results.hourly_unmet_demand)
        full_pumping_power_mw = _full_timeline(model.dispatch_results.hourly_pumping_power)
        full_hourly_geothermal_thermal_output = _full_timeline(model.dispatch_results.hourly_geothermal_thermal_output)
        full_hourly_geothermal_electric_output = _full_timeline(model.dispatch_results.hourly_geothermal_electric_output)
        full_hourly_gross_electric_output = _full_timeline(model.dispatch_results.hourly_gross_electric_output)
        full_hourly_heat_extracted = _full_timeline(model.dispatch_results.hourly_heat_extracted)
        full_hourly_tentering_powerplant = _full_timeline(model.dispatch_results.hourly_tentering_powerplant)
        full_hourly_reinjection_temperature = _full_timeline(model.dispatch_results.hourly_reinjection_temperature)
        full_hourly_availability = _full_timeline(model.dispatch_results.hourly_availability)
        full_hourly_first_law_efficiency = _full_timeline(model.dispatch_results.hourly_first_law_efficiency)
        full_dispatch_demand_mw = _full_timeline(model.dispatch_results.hourly_thermal_demand)

        pumping_power_mw = full_pumping_power_mw

        model.economics.timestepsperyear.value = timesteps_per_year
        model.surfaceplant.utilization_factor.value = float(np.average(full_hourly_runtime_fraction))

        model.wellbores.ProducedTemperature.value = full_hourly_produced_temperature
        model.wellbores.PumpingPower.value = pumping_power_mw.copy()
        model.wellbores.PumpingPowerInj.value = pumping_power_mw.copy()
        model.wellbores.PumpingPowerProd.value = np.zeros_like(pumping_power_mw)
        model.wellbores.redrill.value = 0

        if demand_type == "thermal":
            served_heat_kwh = full_served_demand_kwh
            unmet_demand_kwh = full_unmet_demand_kwh
            model.surfaceplant.HeatProduced.value = (served_heat_kwh / 1000.0).copy()
            model.surfaceplant.HeatExtracted.value = (
                model.surfaceplant.HeatProduced.value / efficiency if efficiency > 0 else np.zeros_like(served_heat_kwh)
            )
            model.surfaceplant.ElectricityProduced.value = np.zeros(total_timesteps)
            model.surfaceplant.NetElectricityProduced.value = np.zeros(total_timesteps)
            model.surfaceplant.TenteringPP.value = np.zeros(total_timesteps)
            model.surfaceplant.Availability.value = np.zeros(total_timesteps)
            model.surfaceplant.FirstLawEfficiency.value = np.zeros(total_timesteps)
        else:
            served_heat_kwh = full_served_demand_kwh
            unmet_demand_kwh = full_unmet_demand_kwh
            model.surfaceplant.HeatProduced.value = np.zeros(total_timesteps)
            model.surfaceplant.HeatExtracted.value = full_hourly_heat_extracted.copy()
            model.surfaceplant.ElectricityProduced.value = full_hourly_gross_electric_output.copy()
            model.surfaceplant.NetElectricityProduced.value = full_hourly_geothermal_electric_output.copy()
            model.surfaceplant.TenteringPP.value = full_hourly_tentering_powerplant.copy()
            model.surfaceplant.Availability.value = full_hourly_availability.copy()
            model.surfaceplant.FirstLawEfficiency.value = full_hourly_first_law_efficiency.copy()

        model.surfaceplant.PumpingkWh.value = np.zeros(plant_lifetime_years)
        model.surfaceplant.HeatkWhExtracted.value = np.zeros(plant_lifetime_years)
        model.surfaceplant.HeatkWhProduced.value = np.zeros(plant_lifetime_years)
        model.surfaceplant.TotalkWhProduced.value = np.zeros(plant_lifetime_years)
        model.surfaceplant.NetkWhProduced.value = np.zeros(plant_lifetime_years)
        model.surfaceplant.RemainingReservoirHeatContent.value = np.zeros(plant_lifetime_years)

        for year_index in range(plant_lifetime_years):
            start = year_index * timesteps_per_year
            end = start + timesteps_per_year
            if demand_type == "thermal":
                model.surfaceplant.HeatkWhProduced.value[year_index] = float(np.sum(served_heat_kwh[start:end]))
            model.surfaceplant.HeatkWhExtracted.value[year_index] = float(np.sum(model.surfaceplant.HeatExtracted.value[start:end] * 1000.0))
            model.surfaceplant.PumpingkWh.value[year_index] = float(np.sum(pumping_power_mw[start:end] * 1000.0))
            model.surfaceplant.TotalkWhProduced.value[year_index] = float(
                np.sum(model.surfaceplant.ElectricityProduced.value[start:end] * 1000.0)
            )
            model.surfaceplant.NetkWhProduced.value[year_index] = float(
                np.sum(model.surfaceplant.NetElectricityProduced.value[start:end] * 1000.0)
            )

        model.surfaceplant.RemainingReservoirHeatContent.value = model.surfaceplant.remaining_reservoir_heat_content(
            model.reserv.InitialReservoirHeatContent.value,
            model.surfaceplant.HeatkWhExtracted.value,
        )
        model.surfaceplant._calculate_derived_outputs(model)

        annual_served_kwh = (
            model.surfaceplant.HeatkWhProduced.value if demand_type == "thermal" else model.surfaceplant.NetkWhProduced.value
        )
        annual_unmet_kwh = np.array([
            float(np.sum(unmet_demand_kwh[year_index * timesteps_per_year:(year_index + 1) * timesteps_per_year]))
            for year_index in range(plant_lifetime_years)
        ])
        dispatch_demand_kwh = full_dispatch_demand_mw * 1000.0 * time_step_hours
        annual_demand_kwh = np.array([
            float(np.sum(dispatch_demand_kwh[year_index * timesteps_per_year:(year_index + 1) * timesteps_per_year]))
            for year_index in range(plant_lifetime_years)
        ])

        analysis_year_slice = slice(analysis_start_year - 1, analysis_end_year - 1)

        model.dispatch_results.hourly_produced_temperature = full_hourly_produced_temperature[
            analysis_start_index:analysis_end_index
        ].copy()
        model.dispatch_results.hourly_flow = full_hourly_flow[analysis_start_index:analysis_end_index].copy()
        model.dispatch_results.hourly_runtime_fraction = full_hourly_runtime_fraction[
            analysis_start_index:analysis_end_index
        ].copy()
        model.dispatch_results.hourly_demand_served = full_served_demand_kwh[analysis_start_index:analysis_end_index].copy()
        model.dispatch_results.hourly_unmet_demand = full_unmet_demand_kwh[analysis_start_index:analysis_end_index].copy()
        model.dispatch_results.hourly_pumping_power = full_pumping_power_mw[analysis_start_index:analysis_end_index].copy()
        model.dispatch_results.hourly_geothermal_thermal_output = full_hourly_geothermal_thermal_output[
            analysis_start_index:analysis_end_index
        ].copy()
        model.dispatch_results.hourly_geothermal_electric_output = full_hourly_geothermal_electric_output[
            analysis_start_index:analysis_end_index
        ].copy()
        model.dispatch_results.hourly_gross_electric_output = full_hourly_gross_electric_output[
            analysis_start_index:analysis_end_index
        ].copy()
        model.dispatch_results.hourly_heat_extracted = full_hourly_heat_extracted[analysis_start_index:analysis_end_index].copy()
        model.dispatch_results.hourly_tentering_powerplant = full_hourly_tentering_powerplant[
            analysis_start_index:analysis_end_index
        ].copy()
        model.dispatch_results.hourly_reinjection_temperature = full_hourly_reinjection_temperature[
            analysis_start_index:analysis_end_index
        ].copy()
        model.dispatch_results.hourly_availability = full_hourly_availability[analysis_start_index:analysis_end_index].copy()
        model.dispatch_results.hourly_first_law_efficiency = full_hourly_first_law_efficiency[
            analysis_start_index:analysis_end_index
        ].copy()
        model.dispatch_results.hourly_thermal_demand = full_dispatch_demand_mw[analysis_start_index:analysis_end_index].copy()

        analysis_served_kwh = annual_served_kwh[analysis_year_slice]
        analysis_unmet_kwh = annual_unmet_kwh[analysis_year_slice]
        analysis_demand_kwh = annual_demand_kwh[analysis_year_slice]
        analysis_hourly_served_kwh = model.dispatch_results.hourly_demand_served
        analysis_hourly_unmet_kwh = model.dispatch_results.hourly_unmet_demand
        analysis_hourly_demand_mw = model.dispatch_results.hourly_thermal_demand

        design_output_mw = model.dispatch_results.summary_metrics.get(
            "design_heat_produced_mw" if demand_type == "thermal" else "design_net_electricity_produced_mw",
            0.0,
        )
        observed_peak_output_mw = float(np.max(analysis_hourly_served_kwh) / 1000.0) if analysis_hourly_served_kwh.size > 0 else 0.0
        capacity_basis_mw = max(design_output_mw, observed_peak_output_mw)
        average_runtime_fraction = float(np.average(model.dispatch_results.hourly_runtime_fraction))
        average_capacity_factor = float(
            np.average(
                analysis_hourly_served_kwh / (capacity_basis_mw * 1000.0)
            )
        ) if capacity_basis_mw > 0 else 0.0

        model.dispatch_results.annual_aggregates = {"analysis_years": list(range(analysis_start_year, analysis_end_year))}
        model.dispatch_results.summary_metrics.update(
            {
                "dispatch_analysis_start_year": float(analysis_start_year),
                "dispatch_analysis_end_year": float(analysis_end_year),
                "dispatch_analysis_year_count": float(analysis_end_year - analysis_start_year),
                "peak_hourly_demand_mw": float(np.max(analysis_hourly_demand_mw)),
                "average_runtime_fraction": average_runtime_fraction,
                "dispatch_capacity_factor": average_capacity_factor,
                "observed_peak_flow_kg_per_sec": float(np.max(model.dispatch_results.hourly_flow)),
            }
        )
        if demand_type == "thermal":
            model.dispatch_results.annual_aggregates.update(
                {
                    "annual_served_heat_kwh": analysis_served_kwh.tolist(),
                    "annual_unmet_heat_kwh": analysis_unmet_kwh.tolist(),
                    "annual_heat_demand_kwh": analysis_demand_kwh.tolist(),
                }
            )
            model.dispatch_results.summary_metrics.update(
                {
                    "peak_served_heat_kwh": float(np.max(analysis_hourly_served_kwh)),
                    "peak_unmet_heat_kwh": float(np.max(analysis_hourly_unmet_kwh)),
                    "annual_served_heat_kwh": float(np.sum(analysis_served_kwh)),
                    "annual_unmet_heat_kwh": float(np.sum(analysis_unmet_kwh)),
                }
            )
        else:
            model.dispatch_results.annual_aggregates.update(
                {
                    "annual_served_electricity_kwh": analysis_served_kwh.tolist(),
                    "annual_unmet_electricity_kwh": analysis_unmet_kwh.tolist(),
                    "annual_electricity_demand_kwh": analysis_demand_kwh.tolist(),
                }
            )
            model.dispatch_results.summary_metrics.update(
                {
                    "peak_served_electricity_kwh": float(np.max(analysis_hourly_served_kwh)),
                    "peak_unmet_electricity_kwh": float(np.max(analysis_hourly_unmet_kwh)),
                    "annual_served_electricity_kwh": float(np.sum(analysis_served_kwh)),
                    "annual_unmet_electricity_kwh": float(np.sum(analysis_unmet_kwh)),
                }
            )


def create_operating_mode_strategy(operating_mode: OperatingMode) -> OperatingModeStrategy:
    if operating_mode == OperatingMode.DISPATCHABLE:
        return DispatchableOperatingModeStrategy()

    return BaseloadOperatingModeStrategy()
