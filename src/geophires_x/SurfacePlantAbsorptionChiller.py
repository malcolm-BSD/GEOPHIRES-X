import numpy as np
from .Parameter import boolParameter, floatParameter, intParameter, listParameter, OutputParameter, strParameter
from .SurfacePlant import SurfacePlant
from .Units import *
import geophires_x.Model as Model


class SurfacePlantAbsorptionChiller(SurfacePlant):
    def __init__(self, model: Model):
        """
        The __init__ function is called automatically when a class is instantiated.
        It initializes the attributes of an object, and sets default values for certain arguments that can be overridden
         by user input.
        The __init__ function is used to set up all the parameters in the Surfaceplant.
        :param model: The container class of the application, giving access to everything else, including the logger
        :type model: :class:`~geophires_x.Model.Model`
        :return: None
        """

        model.logger.info(f"Init {self.__class__.__name__}: {__name__}")
        super().__init__(model)  # Initialize all the parameters in the superclass

        # Set up all the Parameters that will be predefined by this class using the different types of parameter classes.
        # Setting up includes giving it a name, a default value, The Unit Type (length, volume, temperature, etc.) and
        # Unit Name of that value, sets it as required (or not), sets allowable range, the error message if that range
        # is exceeded, the ToolTip Text, and the name of teh class that created it.
        # This includes setting up temporary variables that will be available to all the class but noy read in by user,
        # or used for Output
        # This also includes all Parameters that are calculated and then published using the Printouts function.

        # These dictionaries contain a list of all the parameters set in this object, stored as "Parameter" and
        # "OutputParameter" Objects.  This will allow us later to access them in a user interface and get that list,
        # along with unit type, preferred units, etc.

        self.setinjectionpressurefixed = False
        self.MyClass = self.__class__.__name__
        self.MyPath = __file__

        # Input parameters absorption chiller
        self.absorption_chiller_cop = self.ParameterDict[self.absorption_chiller_cop.Name] = floatParameter(
            "Absorption Chiller COP",
            value=0.7,
            Min=0.1,
            Max=1.5,
            UnitType=Units.PERCENT,
            PreferredUnits=PercentUnit.TENTH, CurrentUnits=PercentUnit.TENTH,
            ErrMessage="assume default absorption chiller COP (0.7)",
            ToolTipText="Specify the coefficient of performance (COP) of the absorption chiller"
        )

        # Opt-in advanced absorption chiller subsystem. When True the new
        # AbsorptionChiller subsystem (in src/geophires_x/absorption) will be
        # used. Default True for the new design; set to False to preserve
        # legacy behaviour exactly.
        self.use_advanced_absorption_chiller = self.ParameterDict["Use Advanced Absorption Chiller"] = boolParameter(
            "Use Advanced Absorption Chiller",
            DefaultValue=True,
            value=True,
            Required=False,
            ErrMessage="Enable advanced absorption chiller subsystem (default: True)",
            ToolTipText="If True use the advanced AbsorptionChiller subsystem (opt-in)."
        )

        # Canonical top-level absorption chiller configuration parameters
        self.absorption_chiller_use_pint = self.ParameterDict["Absorption Chiller Use Pint"] = boolParameter(
            "Absorption Chiller Use Pint",
            DefaultValue=True,
            value=True,
            Required=False,
            ErrMessage="Enable Pint units handling for Absorption Chiller (default: True)",
            ToolTipText="Enable Pint units handling for the Absorption Chiller (recommended).",
        )

        self.absorption_chiller_use_coolprop = self.ParameterDict["Absorption Chiller Use CoolProp"] = boolParameter(
            "Absorption Chiller Use CoolProp",
            DefaultValue=True,
            value=True,
            Required=False,
            ErrMessage="Enable CoolProp for fluid properties (default: True)",
            ToolTipText="Enable CoolProp for fluid properties used by the Absorption Chiller (best-effort).",
        )

        self.absorption_chiller_refrigerant_family = self.ParameterDict["Absorption Chiller Refrigerant Family"] = strParameter(
            "Absorption Chiller Refrigerant Family",
            DefaultValue="LiBr-water",
            Required=False,
            ErrMessage="assume default refrigerant family (LiBr-water)",
            ToolTipText="Refrigerant family for Absorption Chiller (e.g., 'LiBr-water' or 'NH3-water').",
        )

        self.absorption_chiller_effect_type = self.ParameterDict["Absorption Chiller Effect Type"] = strParameter(
            "Absorption Chiller Effect Type",
            DefaultValue="single",
            Required=False,
            ErrMessage="assume default effect type (single)",
            ToolTipText="Absorption chiller effect type: 'single' | 'double' | 'triple'.",
        )

        self.absorption_chiller_effect_multiplier_override = self.ParameterDict["Absorption Chiller Effect Multiplier Override"] = floatParameter(
            "Absorption Chiller Effect Multiplier Override",
            DefaultValue=None,
            value=None,
            Required=False,
            ErrMessage="optional override of effect multiplier (leave blank to use defaults)",
            ToolTipText="Optional override of the effect multiplier used to scale COPs.",
        )

        self.absorption_chiller_min_plr = self.ParameterDict["Absorption Chiller Min Part Load Ratio"] = floatParameter(
            "Absorption Chiller Min Part Load Ratio",
            DefaultValue=0.2,
            Min=0.0,
            Max=1.0,
            Required=False,
            ErrMessage="assume default min part-load ratio (0.2)",
            ToolTipText="Minimum part-load ratio for Absorption Chiller operation.",
        )

        self.absorption_chiller_turndown = self.ParameterDict["Absorption Chiller Turndown Ratio"] = floatParameter(
            "Absorption Chiller Turndown Ratio",
            DefaultValue=3.0,
            Min=1.0,
            Required=False,
            ErrMessage="assume default turndown ratio (3.0)",
            ToolTipText="Turndown ratio for Absorption Chiller (max/min PLR ratio).",
        )

        self.absorption_chiller_chilled_deltaT = self.ParameterDict["Absorption Chiller Chilled DeltaT"] = floatParameter(
            "Absorption Chiller Chilled DeltaT",
            DefaultValue=5.0,
            Min=0.1,
            Required=False,
            ErrMessage="assume default chilled deltaT (5.0 K)",
            ToolTipText="Design chilled-water delta-T used to compute chilled mass flow.",
        )

        self.absorption_chiller_pump_head = self.ParameterDict["Absorption Chiller Pump Head"] = floatParameter(
            "Absorption Chiller Pump Head",
            DefaultValue=30.0,
            Min=0.0,
            Required=False,
            ErrMessage="assume default pump head (30 m)",
            ToolTipText="Pump head used for auxiliary pumping power estimates.",
        )

        self.absorption_chiller_pump_efficiency = self.ParameterDict["Absorption Chiller Pump Efficiency"] = floatParameter(
            "Absorption Chiller Pump Efficiency",
            DefaultValue=0.70,
            Min=0.0,
            Max=1.0,
            Required=False,
            ErrMessage="assume default pump efficiency (0.70)",
            ToolTipText="Pump efficiency used for auxiliary power estimates.",
        )

        self.absorption_chiller_catalog_path = self.ParameterDict["Absorption Chiller Catalog Path"] = strParameter(
            "Absorption Chiller Catalog Path",
            DefaultValue="data/absorption_chiller_catalog_default.csv",
            Required=False,
            ErrMessage="assume default embedded catalog path",
            ToolTipText="Path to user-supplied absorption chiller catalog CSV (optional).",
        )

        # Advanced chiller MILP tuning: number of PLR segments used in piecewise-linear COP approximation
        self.absorption_chiller_dispatch_strategy = self.ParameterDict["Absorption Chiller Dispatch Strategy"] = strParameter(
            "Absorption Chiller Dispatch Strategy",
            DefaultValue="min_cost",
            Required=False,
            ErrMessage="assume default absorption chiller dispatch strategy (min_cost)",
            ToolTipText="Absorption chiller dispatch strategy: 'min_cost' | 'min_units' | 'follow_heat'.",
        )

        self.absorption_chiller_n_segments = self.ParameterDict["Absorption Chiller PLR Segments"] = intParameter(
            "Absorption Chiller PLR Segments",
            DefaultValue=5,
            AllowableRange=list(range(1, 21)),
            UnitType=Units.NONE,
            ErrMessage="number of PLR segments must be >=1",
            ToolTipText="Number of PLR segments used for piecewise-linear COP approximation in MILP dispatch (default 5)",
        )

        # Whether to evaluate segment COPs using per-hour temperatures passed in 'temps'
        self.absorption_chiller_use_hourly_temps = self.ParameterDict[self.absorption_chiller_use_hourly_temps.Name] = boolParameter(
            "Absorption Chiller Use Hourly Temperatures",
            DefaultValue=False,
            value=False,
            Required=False,
            ErrMessage="If True, use hourly absorption chiller temperature profiles when provided",
            ToolTipText="When True, the MILP will evaluate COP segments using the absorption chiller hourly temperature profile parameters.",
        )

        self.absorption_chiller_generator_temperature = self.ParameterDict[
            "Absorption Chiller Generator Temperature Profile"
        ] = listParameter(
            "Absorption Chiller Generator Temperature Profile",
            DefaultValue=[],
            UnitType=Units.TEMPERATURE,
            PreferredUnits=TemperatureUnit.CELSIUS,
            CurrentUnits=TemperatureUnit.CELSIUS,
            AllowExtendedInput=True,
            ErrMessage="assume geothermal production temperature for the absorption chiller generator profile",
            ToolTipText="Hourly absorption chiller generator inlet temperature profile.",
        )

        self.absorption_chiller_evaporator_temperature = self.ParameterDict[
            "Absorption Chiller Evaporator Temperature Profile"
        ] = listParameter(
            "Absorption Chiller Evaporator Temperature Profile",
            DefaultValue=[],
            UnitType=Units.TEMPERATURE,
            PreferredUnits=TemperatureUnit.CELSIUS,
            CurrentUnits=TemperatureUnit.CELSIUS,
            AllowExtendedInput=True,
            ErrMessage="assume default absorption chiller evaporator temperature profile (7 degC)",
            ToolTipText="Hourly absorption chiller evaporator or chilled-water supply temperature profile.",
        )

        self.absorption_chiller_condenser_temperature = self.ParameterDict[
            "Absorption Chiller Condenser Temperature Profile"
        ] = listParameter(
            "Absorption Chiller Condenser Temperature Profile",
            DefaultValue=[],
            UnitType=Units.TEMPERATURE,
            PreferredUnits=TemperatureUnit.CELSIUS,
            CurrentUnits=TemperatureUnit.CELSIUS,
            AllowExtendedInput=True,
            ErrMessage="assume default absorption chiller condenser temperature profile (30 degC)",
            ToolTipText="Hourly absorption chiller condenser or ambient temperature profile.",
        )

        # NOTE: legacy dotted-key aliases (e.g. 'AbsorptionChiller.*') were removed
        # in favour of the project's canonical human-friendly parameter names
        # (e.g. 'Absorption Chiller Use Hourly Temperatures'). If you have input
        # files that still use the dotted-key style, update them to the canonical
        # parameter names in `tests/examples/*.txt` or add your own compatibility
        # mapping at runtime prior to calling `read_parameters`.

        # Output Parameters
        self.cooling_produced = self.OutputParameterDict[self.cooling_produced.Name] = OutputParameter(
            Name="Cooling Produced",
            UnitType=Units.POWER,
            PreferredUnits=PowerUnit.MW,
            CurrentUnits=PowerUnit.MW
        )
        self.cooling_kWh_Produced = self.OutputParameterDict[self.cooling_kWh_Produced.Name] = OutputParameter(
            Name="Annual Cooling Produced",
            UnitType=Units.ENERGYFREQUENCY,
            PreferredUnits=EnergyFrequencyUnit.KWhPERYEAR,
            CurrentUnits=EnergyFrequencyUnit.KWhPERYEAR
        )

        model.logger.info(f"Complete {self.__class__.__name__}: {__name__}")

    def __str__(self):
        return "SurfacePlantAbsorptionChiller"

    @staticmethod
    def _as_series(values) -> np.ndarray:
        series = np.asarray(values, dtype=float)
        if series.ndim == 0:
            series = series.reshape(1)
        elif series.ndim == 2:
            if series.shape[1] < 2:
                raise ValueError("Time-series input must contain time-value pairs.")
            series = series[:, 1]
        elif series.ndim > 1:
            series = series.reshape(-1)
        return series

    def _parameter_series_to_mw(self, parameter) -> np.ndarray:
        series = self._as_series(parameter.value)
        units = getattr(parameter, "CurrentYUnits", getattr(parameter, "CurrentUnits", EnergyUnit.KWH))
        return self._series_to_mw(series, units, time_step_hours=1.0)

    def _temperature_profile(self, parameter, default_value, hours: int, model: Model) -> np.ndarray:
        if getattr(parameter, "value", None) is not None and len(getattr(parameter, "value", [])) > 0:
            series = self._as_series(parameter.value)
            source_name = parameter.Name
            user_provided = True
        else:
            series = self._as_series(default_value)
            source_name = "default temperature profile"
            user_provided = False

        if series.size == 0:
            return np.full(hours, 0.0)
        if series.size == 1:
            return np.full(hours, float(series[0]))
        if series.size == hours:
            return series

        if user_provided:
            model.logger.warning(
                f"{source_name} has {series.size} values, but the absorption chiller cooling demand has {hours}; "
                "using its average value for all dispatch hours."
            )
        return np.full(hours, float(np.mean(series)))

    def _advanced_absorption_enabled(self) -> bool:
        try:
            return bool(self.use_advanced_absorption_chiller.value)
        except Exception:
            return False

    def _advanced_absorption_chiller(self):
        from geophires_x.absorption.absorption_chiller import AbsorptionChiller
        from geophires_x.absorption.catalog import Catalog

        cache_key = (
            str(self.absorption_chiller_catalog_path.value),
            str(self.absorption_chiller_refrigerant_family.value),
            str(self.absorption_chiller_effect_type.value),
            float(self.absorption_chiller_cop.value),
            float(self.absorption_chiller_min_plr.value),
            float(self.absorption_chiller_turndown.value),
            float(self.absorption_chiller_chilled_deltaT.value),
            float(self.absorption_chiller_pump_head.value),
            float(self.absorption_chiller_pump_efficiency.value),
            bool(self.absorption_chiller_use_pint.value),
            bool(self.absorption_chiller_use_coolprop.value),
            self.absorption_chiller_effect_multiplier_override.value,
            int(self.absorption_chiller_n_segments.value),
            bool(self.absorption_chiller_use_hourly_temps.value),
            str(self.absorption_chiller_dispatch_strategy.value),
        )
        cached_key = getattr(self, "_advanced_absorption_chiller_cache_key", None)
        cached_chiller = getattr(self, "_advanced_absorption_chiller_cache", None)
        if cached_key == cache_key and cached_chiller is not None:
            return cached_chiller

        chiller = AbsorptionChiller(
            catalog=Catalog(str(self.absorption_chiller_catalog_path.value)),
            refrigerant_family=str(self.absorption_chiller_refrigerant_family.value),
            effect_type=str(self.absorption_chiller_effect_type.value),
            rated_COP=float(self.absorption_chiller_cop.value),
            min_part_load_ratio=float(self.absorption_chiller_min_plr.value),
            turndown_ratio=float(self.absorption_chiller_turndown.value),
            chilled_deltaT_K=float(self.absorption_chiller_chilled_deltaT.value),
            pump_head_m=float(self.absorption_chiller_pump_head.value),
            pump_efficiency=float(self.absorption_chiller_pump_efficiency.value),
            use_pint=bool(self.absorption_chiller_use_pint.value),
            use_coolprop=bool(self.absorption_chiller_use_coolprop.value),
            effect_multiplier_override=self.absorption_chiller_effect_multiplier_override.value,
            n_segments=int(self.absorption_chiller_n_segments.value),
            use_hourly_temps=bool(self.absorption_chiller_use_hourly_temps.value),
            dispatch_strategy=str(self.absorption_chiller_dispatch_strategy.value),
        )
        setattr(self, "_advanced_absorption_chiller_cache_key", cache_key)
        setattr(self, "_advanced_absorption_chiller_cache", chiller)
        setattr(self, "_absorption_chiller_dispatch_bank", None)
        setattr(self, "_absorption_chiller_dispatch_bank_capacity_kW", 0.0)
        return chiller

    def _cooling_demand_peak_mw(self) -> float:
        cooling_demand = getattr(self, "CoolingDemand", None)
        cache_source = getattr(cooling_demand, "value", None)
        cached_source = getattr(self, "_absorption_chiller_peak_demand_cache_source", None)
        if cached_source is cache_source and hasattr(self, "_absorption_chiller_peak_demand_cache_mw"):
            return self._absorption_chiller_peak_demand_cache_mw

        peak_demand_mw = 0.0
        if cooling_demand is not None and getattr(cooling_demand, "value", None) is not None:
            try:
                if len(cooling_demand.value) > 0:
                    peak_demand_mw = float(np.max(self._parameter_series_to_mw(cooling_demand)))
            except Exception:
                pass

        setattr(self, "_absorption_chiller_peak_demand_cache_source", cache_source)
        setattr(self, "_absorption_chiller_peak_demand_cache_mw", peak_demand_mw)
        return peak_demand_mw

    def _dispatch_temperature_value(self, parameter, default_value: float, timestep_index: int) -> float:
        cache = getattr(self, "_absorption_chiller_dispatch_temperature_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_absorption_chiller_dispatch_temperature_cache", cache)

        cache_key = parameter.Name
        cached_value = cache.get(cache_key)
        source_value = getattr(parameter, "value", None)
        if cached_value is None or cached_value[0] is not source_value:
            if source_value is not None and len(source_value) > 0:
                series = self._as_series(source_value)
            else:
                series = self._as_series(default_value)
            cache[cache_key] = (source_value, series)
        else:
            series = cached_value[1]

        if series.size == 0:
            return float(default_value)
        if series.size == 1:
            return float(series[0])
        return float(series[timestep_index % series.size])

    def advanced_dispatch_output(
        self,
        model: Model,
        cooling_demand_mw: float,
        generator_heat_available_mw: float,
        generator_temperature_c: float,
        timestep_index: int = 0,
    ) -> dict[str, float]:
        peak_cooling_mw = max(self._cooling_demand_peak_mw(), float(cooling_demand_mw), 0.0)
        if peak_cooling_mw <= 0.0 or generator_heat_available_mw <= 0.0:
            return {"cooling_produced_mw": 0.0, "q_gen_mw": 0.0, "cop": 0.0}

        chiller = self._advanced_absorption_chiller()
        required_capacity_kW = peak_cooling_mw * 1000.0
        bank = getattr(self, "_absorption_chiller_dispatch_bank", None)
        bank_capacity_kW = float(getattr(self, "_absorption_chiller_dispatch_bank_capacity_kW", 0.0) or 0.0)
        if bank is None or bank_capacity_kW < required_capacity_kW:
            bank = chiller.build_bank(required_capacity_kW)
            setattr(self, "_absorption_chiller_dispatch_bank", bank)
            setattr(self, "_absorption_chiller_dispatch_bank_capacity_kW", required_capacity_kW)

        t_evap = self._dispatch_temperature_value(
            self.absorption_chiller_evaporator_temperature,
            7.0,
            timestep_index,
        )
        t_cond = self._dispatch_temperature_value(
            self.absorption_chiller_condenser_temperature,
            30.0,
            timestep_index,
        )
        if len(getattr(self.absorption_chiller_generator_temperature, "value", [])) > 0:
            generator_temperature_c = self._dispatch_temperature_value(
                self.absorption_chiller_generator_temperature,
                generator_temperature_c,
                timestep_index,
            )
        temps = {
            "t_gen": np.array([generator_temperature_c], dtype=float),
            "t_evap": np.array([float(t_evap)], dtype=float),
            "t_cond": np.array([float(t_cond)], dtype=float),
        }
        results = bank.dispatch_hourly(
            np.array([float(cooling_demand_mw) * 1000.0], dtype=float),
            generator_heat_available_kW_hourly=np.array([float(generator_heat_available_mw) * 1000.0], dtype=float),
            temps=temps,
            mode="dispatch",
            use_milp=False,
        )
        return {
            "cooling_produced_mw": float(results["cooling_produced_hourly"][0]) / 1000.0,
            "q_gen_mw": float(results["q_gen_hourly"][0]) / 1000.0,
            "cop": float(results["COP_hourly"][0]),
            "pump_power_mw": float(results["pump_power_hourly"][0]) / 1000.0,
        }

    def read_parameters(self, model: Model) -> None:
        """
        The read_parameters function reads in the parameters from a dictionary and stores them in the parameters.
        It also handles special cases that need to be handled after a value has been read in and checked.
        If you choose to subclass this master class, you can also choose to override this method (or not), and if you do
        :param model: The container class of the application, giving access to everything else, including the logger
        :return: None
        """
        model.logger.info(f"Init {self.__class__.__name__}: {__name__}")
        super().read_parameters(model)  # Read in all the parameters from the superclass

        # Since there are no parameters that require unique adjustments in this class, we don't need to do anything.

        model.logger.info(f"complete {self.__class__.__name__}: {__name__}")

    def Calculate(self, model: Model) -> None:
        """
        The Calculate function is where all the calculations are done.

        Note about the advanced absorption chiller subsystem:
        - If the parameter ``Use Advanced Absorption Chiller`` is True (default),
          this method will call into the new :mod:`geophires_x.absorption`
          subsystem. That subsystem will attempt to size and dispatch
          commercial absorption chillers; when PuLP is available the catalog
          selection uses integer programming and the per-hour dispatch will
          also try to solve a small integer program to minimize cost.
        - If ``Use Advanced Absorption Chiller`` is False or if the advanced
          subsystem fails, the method falls back to the legacy, scalar
          calculation that preserves previous behaviour.

        The advanced dispatch supports strategies including ``min_cost``,
        ``min_units`` and ``follow_heat`` (see the documentation in
        ``src/geophires_x/absorption/chiller_bank.py``).

        :param model: The container class of the application, giving access to everything else, including the logger
        :type model: :class:`~geophires_x.Model.Model`
        :return: Nothing, but it does make calculations and set values in the model
        """
        model.logger.info(f"Init {self.__class__.__name__}: {__name__}")

        # This is where all the calculations are made using all the values that have been set.
        # If you subclass this class, you can choose to run these calculations before (or after) your calculations,
        # but that assumes you have set all the values that are required for these calculations
        # If you choose to subclass this master class, you can also choose to override this method (or not),
        # and if you do, do it before or after you call you own version of this method.  If you do, you can also choose
        # to call this method from you class, which can effectively run the calculations of the superclass, making all
        # the values available to your methods. but you had better have set all the parameters!

        # calculate produced electricity/direct-use heat
        # absorption chiller: we don't consider end-use efficiency factor here.
        # All extracted heat will go to absorption chiller and there is the end-use efficiency factor. [MWth]
        self.HeatExtracted.value = model.wellbores.nprod.value * model.wellbores.prodwellflowrate.value * model.reserv.cpwater.value * (
            model.wellbores.ProducedTemperature.value - model.wellbores.Tinj.value) / 1E6  # heat extracted from geofluid [MWth]
        self.HeatProduced.value = self.HeatExtracted.value

        # Use advanced AbsorptionChiller subsystem when enabled; otherwise keep legacy behavior
        try:
            use_adv = bool(getattr(self.use_advanced_absorption_chiller, "value", False))
        except Exception:
            use_adv = False

        if use_adv:
            try:
                # Import here to avoid import-time dependency if not used
                from geophires_x.absorption.absorption_chiller import AbsorptionChiller
                from geophires_x.absorption.catalog import Catalog

                ch = AbsorptionChiller(
                    catalog=Catalog(str(self.absorption_chiller_catalog_path.value)),
                    refrigerant_family=str(self.absorption_chiller_refrigerant_family.value),
                    effect_type=str(self.absorption_chiller_effect_type.value),
                    rated_COP=float(self.absorption_chiller_cop.value),
                    min_part_load_ratio=float(self.absorption_chiller_min_plr.value),
                    turndown_ratio=float(self.absorption_chiller_turndown.value),
                    chilled_deltaT_K=float(self.absorption_chiller_chilled_deltaT.value),
                    pump_head_m=float(self.absorption_chiller_pump_head.value),
                    pump_efficiency=float(self.absorption_chiller_pump_efficiency.value),
                    use_pint=bool(self.absorption_chiller_use_pint.value),
                    use_coolprop=bool(self.absorption_chiller_use_coolprop.value),
                    effect_multiplier_override=self.absorption_chiller_effect_multiplier_override.value,
                    n_segments=int(self.absorption_chiller_n_segments.value),
                    use_hourly_temps=bool(self.absorption_chiller_use_hourly_temps.value),
                    dispatch_strategy=str(self.absorption_chiller_dispatch_strategy.value),
                )

                # Determine cooling demand time series in GEOPHIRES units (MW).
                cooling_series_mw = None
                try:
                    cd = getattr(self, "CoolingDemand", None)
                    if cd is not None and getattr(cd, "value", None) is not None and len(getattr(cd, "value")) > 0:
                        # convert user series to MW if necessary
                        try:
                            cooling_series_mw = self._parameter_series_to_mw(cd)
                        except Exception:
                            cooling_series_mw = self._as_series(cd.value)
                except Exception:
                    cooling_series_mw = None

                if cooling_series_mw is None:
                    # Fallback: derive cooling demand from available heat (legacy-equivalent)
                    cooling_series_mw = (
                        np.asarray(self.HeatProduced.value, dtype=float)
                        * float(self.absorption_chiller_cop.value)
                        * float(self.enduse_efficiency_factor.value)
                    )

                hours = len(cooling_series_mw)
                t_gen = self._temperature_profile(
                    self.absorption_chiller_generator_temperature,
                    model.wellbores.ProducedTemperature.value,
                    hours,
                    model,
                )
                t_evap = self._temperature_profile(
                    self.absorption_chiller_evaporator_temperature,
                    7.0,
                    hours,
                    model,
                )
                t_cond = self._temperature_profile(
                    self.absorption_chiller_condenser_temperature,
                    30.0,
                    hours,
                    model,
                )
                temps = {"t_gen": t_gen, "t_evap": t_evap, "t_cond": t_cond}

                # choose mode based on operating_mode parameter
                op_mode = getattr(self.operating_mode, "value", "").__str__() if hasattr(self.operating_mode, "value") else str(getattr(self.operating_mode, "value", ""))
                mode = "dispatch" if str(op_mode).lower().startswith("dispatch") else "baseload"

                cooling_series_kw = np.asarray(cooling_series_mw, dtype=float) * 1000.0
                results = ch.evaluate_hourly(
                    cooling_series_kw,
                    t_gen,
                    chilled_supply_setpoint_c=7.0,
                    ambient_temp_hourly=t_cond,
                    temps=temps,
                    mode=mode,
                    use_milp=False,
                )

                # store key outputs into SurfacePlant outputs
                cooling_produced_kw = results.get("cooling_produced_hourly", cooling_series_kw)
                self.cooling_produced.value = np.asarray(cooling_produced_kw, dtype=float) / 1000.0
                # store additional chiller outputs for downstream use
                setattr(self, "_absorption_chiller_results", results)
            except Exception as exc:  # pragma: no cover - liberal fallback
                model.logger.exception("Advanced AbsorptionChiller failed; falling back to legacy calculation: %s", exc)
                self.cooling_produced.value = self.HeatProduced.value * self.absorption_chiller_cop.value * self.enduse_efficiency_factor.value
        else:
            self.cooling_produced.value = self.HeatProduced.value * self.absorption_chiller_cop.value * self.enduse_efficiency_factor.value  # MW

        # Calculate annual electricity/heat production
        # all end-use options have "heat extracted from reservoir" and pumping kWs
        self.HeatkWhExtracted.value = np.zeros(self.plant_lifetime.value)
        self.PumpingkWh.value = np.zeros(self.plant_lifetime.value)

        def _integrate_slice(series, _i):
            return SurfacePlant.integrate_time_series_slice(
                series, _i, model.economics.timestepsperyear.value, self.utilization_factor.value
            )

        for i in range(0, self.plant_lifetime.value):
            self.HeatkWhExtracted.value[i] = _integrate_slice(self.HeatExtracted.value, i)
            self.PumpingkWh.value[i] = _integrate_slice(model.wellbores.PumpingPower.value, i)

        self.HeatkWhProduced.value = np.zeros(self.plant_lifetime.value)
        for i in range(0, self.plant_lifetime.value):
            self.HeatkWhProduced.value[i] = _integrate_slice(self.HeatProduced.value, i)

        self.cooling_kWh_Produced.value = np.zeros(self.plant_lifetime.value)
        for i in range(0, self.plant_lifetime.value):
            self.cooling_kWh_Produced.value[i] = _integrate_slice(self.cooling_produced.value, i)

        # calculate reservoir heat content
        self.RemainingReservoirHeatContent.value = SurfacePlant.remaining_reservoir_heat_content(
            self, model.reserv.InitialReservoirHeatContent.value, self.HeatkWhExtracted.value)

        self._calculate_derived_outputs(model)
        model.logger.info(f"complete {self.__class__.__name__}: {__name__}")
