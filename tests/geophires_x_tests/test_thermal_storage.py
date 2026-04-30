import unittest

from geophires_x.GeoPHIRESUtils import density_water_kg_per_m3
from geophires_x.GeoPHIRESUtils import heat_capacity_water_J_per_kg_per_K
from geophires_x.GeoPHIRESUtils import quantity
from geophires_x.ThermalStorage import JOULES_PER_MWH
from geophires_x.ThermalStorage import ThermalStorageModel


class ThermalStorageTestCase(unittest.TestCase):
    def _new_storage(self, **overrides) -> ThermalStorageModel:
        kwargs = {
            "volume_m3": 1000.0,
            "minimum_temperature_c": 120.0,
            "maximum_temperature_c": 160.0,
            "initial_temperature_c": 140.0,
            "pressure_mpa": 1.0,
            "charge_efficiency": 0.98,
            "discharge_efficiency": 0.98,
            "daily_heat_loss_fraction": 0.005,
        }
        kwargs.update(overrides)
        return ThermalStorageModel(**kwargs)

    def test_capacity_and_state_of_charge_use_water_properties(self):
        storage = self._new_storage()
        reference_temperature_c = 140.0
        pressure = quantity(1.0, "MPa")
        expected_capacity_mwh = (
            density_water_kg_per_m3(reference_temperature_c, pressure=pressure)
            * 1000.0
            * heat_capacity_water_J_per_kg_per_K(reference_temperature_c, pressure=pressure)
            * 40.0
            / JOULES_PER_MWH
        )

        self.assertAlmostEqual(expected_capacity_mwh, storage.usable_capacity_mwh, places=8)
        self.assertAlmostEqual(0.5, storage.soc_fraction, places=8)
        self.assertAlmostEqual(140.0, storage.temperature_c, places=8)
        self.assertAlmostEqual(storage.usable_capacity_mwh * 0.5, storage.stored_energy_mwh, places=8)

    def test_auto_pressure_uses_required_liquid_pressure(self):
        storage = self._new_storage(pressure_mpa=None, maximum_temperature_c=180.0)
        expected_pressure = ThermalStorageModel.required_liquid_pressure_mpa(180.0, 1.10)

        self.assertAlmostEqual(expected_pressure, storage.pressure_mpa, places=8)

    def test_pressure_below_saturation_requirement_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "TESS Pressure"):
            self._new_storage(pressure_mpa=0.1, maximum_temperature_c=180.0)

    def test_discharge_is_limited_by_power_limit(self):
        storage = self._new_storage(initial_temperature_c=160.0, maximum_discharge_power_mw=5.0)

        result = storage.discharge(demand_mw=12.0, dt_hours=1.0)

        self.assertAlmostEqual(5.0, result.discharged_to_load_mw)
        self.assertAlmostEqual(7.0, result.unmet_demand_mw)
        self.assertAlmostEqual(5.0 / 0.98, result.storage_energy_draw_mwh)
        self.assertAlmostEqual(storage.usable_capacity_mwh - (5.0 / 0.98), storage.stored_energy_mwh)

    def test_discharge_is_limited_by_stored_energy(self):
        storage = self._new_storage(initial_temperature_c=120.0)

        result = storage.discharge(demand_mw=10.0, dt_hours=1.0)

        self.assertEqual(0.0, result.discharged_to_load_mw)
        self.assertEqual(10.0, result.unmet_demand_mw)
        self.assertEqual(0.0, storage.stored_energy_mwh)

    def test_charge_is_limited_by_power_limit(self):
        storage = self._new_storage(
            initial_temperature_c=120.0,
            charge_efficiency=0.90,
            maximum_charge_power_mw=3.0,
        )

        result = storage.charge(geothermal_charge_available_mw=10.0, dt_hours=1.0, source_temperature_c=180.0)

        self.assertAlmostEqual(3.0, result.geothermal_charge_accepted_mw)
        self.assertAlmostEqual(7.0, result.curtailed_charge_mw)
        self.assertAlmostEqual(2.7, result.stored_energy_added_mwh)
        self.assertAlmostEqual(2.7, storage.stored_energy_mwh)
        self.assertAlmostEqual(0.3, result.charge_efficiency_loss_mw)

    def test_charge_is_limited_by_remaining_capacity(self):
        storage = self._new_storage(
            initial_temperature_c=160.0,
            charge_efficiency=0.90,
        )

        result = storage.charge(geothermal_charge_available_mw=10.0, dt_hours=1.0, source_temperature_c=180.0)

        self.assertEqual(0.0, result.geothermal_charge_accepted_mw)
        self.assertEqual(10.0, result.curtailed_charge_mw)
        self.assertAlmostEqual(storage.usable_capacity_mwh, storage.stored_energy_mwh)

    def test_charge_rejects_source_temperature_below_tank_temperature(self):
        storage = self._new_storage(initial_temperature_c=150.0)

        result = storage.charge(geothermal_charge_available_mw=10.0, dt_hours=1.0, source_temperature_c=140.0)

        self.assertEqual(0.0, result.geothermal_charge_accepted_mw)
        self.assertEqual(10.0, result.curtailed_charge_mw)

    def test_standby_losses_reduce_stored_energy(self):
        storage = self._new_storage(initial_temperature_c=160.0, daily_heat_loss_fraction=0.24)
        starting_energy = storage.stored_energy_mwh

        result = storage.apply_losses(dt_hours=1.0)

        expected_loss_mwh = starting_energy * 0.24 / 24.0
        self.assertAlmostEqual(expected_loss_mwh, result.standby_loss_mwh)
        self.assertAlmostEqual(expected_loss_mwh, result.standby_loss_mw)
        self.assertAlmostEqual(starting_energy - expected_loss_mwh, storage.stored_energy_mwh)

    def test_step_closes_energy_and_demand_balances(self):
        storage = self._new_storage(
            initial_temperature_c=140.0,
            charge_efficiency=0.90,
            discharge_efficiency=0.80,
            daily_heat_loss_fraction=0.024,
        )

        result = storage.step(
            demand_mw=2.0,
            geothermal_charge_available_mw=5.0,
            dt_hours=1.0,
            source_temperature_c=180.0,
        )

        self.assertAlmostEqual(result.demand_mw, result.discharged_to_load_mw + result.unmet_demand_mw)
        self.assertAlmostEqual(0.0, result.storage_energy_balance_error_mwh, places=10)
        self.assertAlmostEqual(
            result.ending_stored_energy_mwh - result.starting_stored_energy_mwh,
            result.stored_energy_delta_mwh,
            places=10,
        )

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "TESS Volume"):
            self._new_storage(volume_m3=0.0)
        with self.assertRaisesRegex(ValueError, "TESS Maximum Temperature"):
            self._new_storage(minimum_temperature_c=160.0, maximum_temperature_c=120.0)
        with self.assertRaisesRegex(ValueError, "TESS Charge Efficiency"):
            self._new_storage(charge_efficiency=0.0)
