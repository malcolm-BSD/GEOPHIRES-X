import numpy as np
from .Parameter import floatParameter, OutputParameter, boolParameter, strParameter, intParameter
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
        self.Use Advanced Absorption Chiller = self.ParameterDict[self.Use Advanced Absorption Chiller.Name] = boolParameter(
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
        self.absorption_chiller_n_segments = self.ParameterDict[self.absorption_chiller_n_segments.Name] = floatParameter(
            "Absorption Chiller PLR Segments",
            value=5,
            Min=1,
            Max=20,
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
            ErrMessage="If True, provide hourly 'temps' arrays to dispatch_hourly/evaluate_hourly",
            ToolTipText="When True, the MILP will evaluate COP segments using per-hour temperatures supplied to the dispatcher.",
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
            use_adv = bool(getattr(self.Use Advanced Absorption Chiller, "value", False))
        except Exception:
            use_adv = False

        if use_adv:
            try:
                # Import here to avoid import-time dependency if not used
                from geophires_x.absorption.absorption_chiller import AbsorptionChiller

                ch = AbsorptionChiller()

                # Determine cooling demand time series: prefer user-provided CoolingDemand, else derive from available heat
                cooling_series = None
                try:
                    cd = getattr(self, "CoolingDemand", None)
                    if cd is not None and getattr(cd, "value", None) is not None and len(getattr(cd, "value")) > 0:
                        # convert user series to MW if necessary
                        try:
                            cooling_series = self._series_to_mw(cd.value, getattr(cd, "CurrentYUnits", None), time_step_hours=1.0)
                        except Exception:
                            cooling_series = np.asarray(cd.value, dtype=float)
                except Exception:
                    cooling_series = None

                if cooling_series is None:
                    # Fallback: derive cooling demand from available heat (legacy-equivalent)
                    cooling_series = np.asarray(self.HeatProduced.value, dtype=float) * float(self.absorption_chiller_cop.value) * float(self.enduse_efficiency_factor.value)

                # choose mode based on operating_mode parameter
                op_mode = getattr(self.operating_mode, "value", "").__str__() if hasattr(self.operating_mode, "value") else str(getattr(self.operating_mode, "value", ""))
                mode = "dispatch" if str(op_mode).lower().startswith("dispatch") else "baseload"

                results = ch.evaluate_hourly(cooling_series, model.wellbores.ProducedTemperature.value, chilled_supply_setpoint_c=7.0, ambient_temp_hourly=None, mode=mode)

                # store key outputs into SurfacePlant outputs
                self.cooling_produced.value = results.get("cooling_produced_hourly", np.asarray(cooling_series, dtype=float))
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
