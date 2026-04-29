from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from typing import Tuple

from geophires_x.GeoPHIRESUtils import density_water_kg_per_m3
from geophires_x.GeoPHIRESUtils import heat_capacity_water_J_per_kg_per_K
from geophires_x.GeoPHIRESUtils import quantity
from geophires_x.GeoPHIRESUtils import saturation_pressure_water_MPa

JOULES_PER_MWH = 3.6e9


@dataclass(frozen=True)
class ThermalStorageState:
    temperature_c: float
    stored_energy_mwh: float
    usable_capacity_mwh: float
    soc_fraction: float
    available_discharge_mwh: float
    remaining_charge_capacity_mwh: float


@dataclass(frozen=True)
class ThermalStorageDischargeResult:
    demand_mw: float
    discharged_to_load_mw: float
    unmet_demand_mw: float
    storage_energy_draw_mwh: float
    discharge_efficiency_loss_mw: float


@dataclass(frozen=True)
class ThermalStorageChargeResult:
    geothermal_charge_available_mw: float
    geothermal_charge_accepted_mw: float
    curtailed_charge_mw: float
    stored_energy_added_mwh: float
    charge_efficiency_loss_mw: float


@dataclass(frozen=True)
class ThermalStorageLossResult:
    standby_loss_mw: float
    standby_loss_mwh: float


@dataclass(frozen=True)
class ThermalStorageTimestepResult:
    starting_temperature_c: float
    ending_temperature_c: float
    starting_soc_fraction: float
    ending_soc_fraction: float
    starting_stored_energy_mwh: float
    ending_stored_energy_mwh: float
    demand_mw: float
    discharged_to_load_mw: float
    unmet_demand_mw: float
    geothermal_charge_available_mw: float
    geothermal_charge_accepted_mw: float
    curtailed_charge_mw: float
    standby_loss_mw: float
    charge_efficiency_loss_mw: float
    discharge_efficiency_loss_mw: float
    stored_energy_delta_mwh: float
    storage_energy_balance_error_mwh: float


class ThermalStorageModel:
    def __init__(
        self,
        volume_m3: float,
        minimum_temperature_c: float,
        maximum_temperature_c: float,
        initial_temperature_c: float,
        pressure_mpa: Optional[float] = None,
        pressure_safety_factor: float = 1.10,
        charge_efficiency: float = 0.98,
        discharge_efficiency: float = 0.98,
        daily_heat_loss_fraction: float = 0.005,
        maximum_charge_power_mw: Optional[float] = None,
        maximum_discharge_power_mw: Optional[float] = None,
    ):
        self.volume_m3 = self._require_nonnegative("TESS Volume", volume_m3, strictly_positive=True)
        self.minimum_temperature_c = float(minimum_temperature_c)
        self.maximum_temperature_c = float(maximum_temperature_c)
        self.initial_temperature_c = float(initial_temperature_c)
        self.pressure_safety_factor = self._require_nonnegative(
            "TESS Pressure Safety Factor",
            pressure_safety_factor,
            strictly_positive=True,
        )
        self.charge_efficiency = self._require_fraction("TESS Charge Efficiency", charge_efficiency)
        self.discharge_efficiency = self._require_fraction("TESS Discharge Efficiency", discharge_efficiency)
        self.daily_heat_loss_fraction = self._require_nonnegative(
            "TESS Daily Heat Loss Fraction",
            daily_heat_loss_fraction,
        )
        self.maximum_charge_power_mw = self._normalize_optional_power_limit(maximum_charge_power_mw)
        self.maximum_discharge_power_mw = self._normalize_optional_power_limit(maximum_discharge_power_mw)

        if self.maximum_temperature_c <= self.minimum_temperature_c:
            raise ValueError("TESS Maximum Temperature must be greater than TESS Minimum Useful Temperature.")
        if not self.minimum_temperature_c <= self.initial_temperature_c <= self.maximum_temperature_c:
            raise ValueError(
                "TESS Initial Temperature must be between TESS Minimum Useful Temperature and TESS Maximum Temperature."
            )

        required_pressure_mpa = self.required_liquid_pressure_mpa(
            self.maximum_temperature_c,
            self.pressure_safety_factor,
        )
        if pressure_mpa is None:
            self.pressure_mpa = required_pressure_mpa
        else:
            self.pressure_mpa = float(pressure_mpa)
            if self.pressure_mpa < required_pressure_mpa:
                raise ValueError(
                    f"TESS Pressure ({self.pressure_mpa:.3f} MPa) must be at least "
                    f"{required_pressure_mpa:.3f} MPa to keep water liquid at TESS Maximum Temperature."
                )

        reference_temperature_c = (self.minimum_temperature_c + self.maximum_temperature_c) / 2.0
        pressure = quantity(self.pressure_mpa, "MPa")
        self.density_kg_per_m3 = density_water_kg_per_m3(reference_temperature_c, pressure=pressure)
        self.heat_capacity_j_per_kg_k = heat_capacity_water_J_per_kg_per_K(
            reference_temperature_c,
            pressure=pressure,
        )
        self.mass_kg = self.density_kg_per_m3 * self.volume_m3
        self.usable_capacity_mwh = (
            self.mass_kg
            * self.heat_capacity_j_per_kg_k
            * (self.maximum_temperature_c - self.minimum_temperature_c)
            / JOULES_PER_MWH
        )
        if self.usable_capacity_mwh <= 0:
            raise ValueError("TESS usable capacity must be positive.")

        self._stored_energy_mwh = self._energy_from_temperature(self.initial_temperature_c)

    @staticmethod
    def required_liquid_pressure_mpa(maximum_temperature_c: float, pressure_safety_factor: float = 1.10) -> float:
        if pressure_safety_factor < 1.0:
            raise ValueError("TESS Pressure Safety Factor must be greater than or equal to 1.0.")
        return saturation_pressure_water_MPa(float(maximum_temperature_c)) * float(pressure_safety_factor)

    @staticmethod
    def _require_nonnegative(name: str, value: float, strictly_positive: bool = False) -> float:
        value = float(value)
        if strictly_positive and value <= 0.0:
            raise ValueError(f"{name} must be greater than zero.")
        if not strictly_positive and value < 0.0:
            raise ValueError(f"{name} must be greater than or equal to zero.")
        return value

    @staticmethod
    def _require_fraction(name: str, value: float) -> float:
        value = float(value)
        if value <= 0.0 or value > 1.0:
            raise ValueError(f"{name} must be greater than zero and less than or equal to one.")
        return value

    @staticmethod
    def _normalize_optional_power_limit(value: Optional[float]) -> Optional[float]:
        if value is None or float(value) < 0.0:
            return None
        return float(value)

    def _energy_from_temperature(self, temperature_c: float) -> float:
        stored_energy_mwh = (
            self.mass_kg
            * self.heat_capacity_j_per_kg_k
            * (float(temperature_c) - self.minimum_temperature_c)
            / JOULES_PER_MWH
        )
        return self._clip_energy(stored_energy_mwh)

    def _temperature_from_energy(self, stored_energy_mwh: float) -> float:
        return self.minimum_temperature_c + (
            self._clip_energy(stored_energy_mwh) / self.usable_capacity_mwh
        ) * (self.maximum_temperature_c - self.minimum_temperature_c)

    def _clip_energy(self, stored_energy_mwh: float) -> float:
        return min(max(float(stored_energy_mwh), 0.0), self.usable_capacity_mwh)

    def _set_stored_energy(self, stored_energy_mwh: float) -> None:
        self._stored_energy_mwh = self._clip_energy(stored_energy_mwh)

    @staticmethod
    def _validate_power_and_duration(power_mw: float, dt_hours: float, label: str) -> Tuple[float, float]:
        power_mw = float(power_mw)
        dt_hours = float(dt_hours)
        if power_mw < 0.0:
            raise ValueError(f"{label} must be greater than or equal to zero.")
        if dt_hours <= 0.0:
            raise ValueError("TESS timestep duration must be greater than zero.")
        return power_mw, dt_hours

    @property
    def stored_energy_mwh(self) -> float:
        return self._stored_energy_mwh

    @property
    def temperature_c(self) -> float:
        return self._temperature_from_energy(self._stored_energy_mwh)

    @property
    def soc_fraction(self) -> float:
        return self._stored_energy_mwh / self.usable_capacity_mwh if self.usable_capacity_mwh > 0 else 0.0

    @property
    def state(self) -> ThermalStorageState:
        return ThermalStorageState(
            temperature_c=self.temperature_c,
            stored_energy_mwh=self.stored_energy_mwh,
            usable_capacity_mwh=self.usable_capacity_mwh,
            soc_fraction=self.soc_fraction,
            available_discharge_mwh=self.stored_energy_mwh * self.discharge_efficiency,
            remaining_charge_capacity_mwh=self.usable_capacity_mwh - self.stored_energy_mwh,
        )

    def discharge(self, demand_mw: float, dt_hours: float = 1.0) -> ThermalStorageDischargeResult:
        demand_mw, dt_hours = self._validate_power_and_duration(demand_mw, dt_hours, "TESS demand")

        power_limited_demand_mw = demand_mw
        if self.maximum_discharge_power_mw is not None:
            power_limited_demand_mw = min(power_limited_demand_mw, self.maximum_discharge_power_mw)

        stored_energy_limited_delivery_mw = self.stored_energy_mwh * self.discharge_efficiency / dt_hours
        discharged_to_load_mw = min(power_limited_demand_mw, stored_energy_limited_delivery_mw)
        storage_energy_draw_mwh = discharged_to_load_mw * dt_hours / self.discharge_efficiency
        discharge_efficiency_loss_mwh = storage_energy_draw_mwh - discharged_to_load_mw * dt_hours

        self._set_stored_energy(self.stored_energy_mwh - storage_energy_draw_mwh)

        return ThermalStorageDischargeResult(
            demand_mw=demand_mw,
            discharged_to_load_mw=discharged_to_load_mw,
            unmet_demand_mw=max(demand_mw - discharged_to_load_mw, 0.0),
            storage_energy_draw_mwh=storage_energy_draw_mwh,
            discharge_efficiency_loss_mw=discharge_efficiency_loss_mwh / dt_hours,
        )

    def charge(
        self,
        geothermal_charge_available_mw: float,
        dt_hours: float = 1.0,
        source_temperature_c: Optional[float] = None,
    ) -> ThermalStorageChargeResult:
        geothermal_charge_available_mw, dt_hours = self._validate_power_and_duration(
            geothermal_charge_available_mw,
            dt_hours,
            "TESS geothermal charge",
        )

        charge_candidate_mw = geothermal_charge_available_mw
        if source_temperature_c is not None and float(source_temperature_c) <= self.temperature_c:
            charge_candidate_mw = 0.0
        if self.maximum_charge_power_mw is not None:
            charge_candidate_mw = min(charge_candidate_mw, self.maximum_charge_power_mw)

        remaining_capacity_mwh = self.usable_capacity_mwh - self.stored_energy_mwh
        capacity_limited_charge_mw = remaining_capacity_mwh / (self.charge_efficiency * dt_hours)
        geothermal_charge_accepted_mw = min(charge_candidate_mw, max(capacity_limited_charge_mw, 0.0))
        stored_energy_added_mwh = geothermal_charge_accepted_mw * self.charge_efficiency * dt_hours
        charge_efficiency_loss_mwh = geothermal_charge_accepted_mw * (1.0 - self.charge_efficiency) * dt_hours

        self._set_stored_energy(self.stored_energy_mwh + stored_energy_added_mwh)

        return ThermalStorageChargeResult(
            geothermal_charge_available_mw=geothermal_charge_available_mw,
            geothermal_charge_accepted_mw=geothermal_charge_accepted_mw,
            curtailed_charge_mw=max(geothermal_charge_available_mw - geothermal_charge_accepted_mw, 0.0),
            stored_energy_added_mwh=stored_energy_added_mwh,
            charge_efficiency_loss_mw=charge_efficiency_loss_mwh / dt_hours,
        )

    def apply_losses(self, dt_hours: float = 1.0) -> ThermalStorageLossResult:
        _, dt_hours = self._validate_power_and_duration(0.0, dt_hours, "TESS standby loss")
        standby_loss_mwh = min(
            self.stored_energy_mwh * self.daily_heat_loss_fraction * dt_hours / 24.0,
            self.stored_energy_mwh,
        )
        self._set_stored_energy(self.stored_energy_mwh - standby_loss_mwh)
        return ThermalStorageLossResult(
            standby_loss_mw=standby_loss_mwh / dt_hours,
            standby_loss_mwh=standby_loss_mwh,
        )

    def step(
        self,
        demand_mw: float,
        geothermal_charge_available_mw: float,
        dt_hours: float = 1.0,
        source_temperature_c: Optional[float] = None,
        apply_standby_loss: bool = True,
    ) -> ThermalStorageTimestepResult:
        starting_state = self.state

        discharge_result = self.discharge(demand_mw, dt_hours=dt_hours)
        charge_result = self.charge(
            geothermal_charge_available_mw,
            dt_hours=dt_hours,
            source_temperature_c=source_temperature_c,
        )
        loss_result = (
            self.apply_losses(dt_hours=dt_hours)
            if apply_standby_loss
            else ThermalStorageLossResult(standby_loss_mw=0.0, standby_loss_mwh=0.0)
        )
        ending_state = self.state

        stored_energy_delta_mwh = ending_state.stored_energy_mwh - starting_state.stored_energy_mwh
        expected_delta_mwh = (
            charge_result.stored_energy_added_mwh
            - discharge_result.storage_energy_draw_mwh
            - loss_result.standby_loss_mwh
        )

        return ThermalStorageTimestepResult(
            starting_temperature_c=starting_state.temperature_c,
            ending_temperature_c=ending_state.temperature_c,
            starting_soc_fraction=starting_state.soc_fraction,
            ending_soc_fraction=ending_state.soc_fraction,
            starting_stored_energy_mwh=starting_state.stored_energy_mwh,
            ending_stored_energy_mwh=ending_state.stored_energy_mwh,
            demand_mw=discharge_result.demand_mw,
            discharged_to_load_mw=discharge_result.discharged_to_load_mw,
            unmet_demand_mw=discharge_result.unmet_demand_mw,
            geothermal_charge_available_mw=charge_result.geothermal_charge_available_mw,
            geothermal_charge_accepted_mw=charge_result.geothermal_charge_accepted_mw,
            curtailed_charge_mw=charge_result.curtailed_charge_mw,
            standby_loss_mw=loss_result.standby_loss_mw,
            charge_efficiency_loss_mw=charge_result.charge_efficiency_loss_mw,
            discharge_efficiency_loss_mw=discharge_result.discharge_efficiency_loss_mw,
            stored_energy_delta_mwh=stored_energy_delta_mwh,
            storage_energy_balance_error_mwh=stored_energy_delta_mwh - expected_delta_mwh,
        )
