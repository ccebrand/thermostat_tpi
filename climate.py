"""Adds support for thermostat tpi units."""
from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

import voluptuous as vol
from datetime import timedelta, datetime

from homeassistant.components.climate import (
    ATTR_PRESET_MODE,
    PLATFORM_SCHEMA,
    PRESET_ACTIVITY,
    PRESET_AWAY,
    PRESET_COMFORT,
    PRESET_HOME,
    PRESET_NONE,
    PRESET_SLEEP,
    ClimateEntity,
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
)
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_TEMPERATURE,
    CONF_NAME,
    CONF_UNIQUE_ID,
    EVENT_HOMEASSISTANT_START,
    PRECISION_HALVES,
    PRECISION_TENTHS,
    PRECISION_WHOLE,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_ON,
    STATE_OFF,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    STATE_HOME,
)
from homeassistant.core import DOMAIN as HA_DOMAIN, CoreState, HomeAssistant, callback
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
    async_call_later,
)
from homeassistant.helpers.reload import async_setup_reload_service
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

from . import DOMAIN, PLATFORMS
from homeassistant.helpers.config_validation import boolean

_LOGGER = logging.getLogger(__name__)

DEFAULT_TOLERANCE = 0.3
DEFAULT_NAME = "Thermostat TPI"
DEFAULT_T_COEFF = 0.01
DEFAULT_C_COEFF = 0.6
DEFAULT_TARGET_TEMPERATURE = 20

CONF_HEATER = "heater"
CONF_IN_TEMP_SENSOR = "in_temperature_sensor"
CONF_OUT_TEMP_SENSOR = "out_temperature_sensor"
CONF_TARGET_TEMP = "target_temp"
CONF_MIN_TEMP = "min_temp"
CONF_AC_MODE = "ac_mode"
CONF_C_COEFF = "c_coefficient"
CONF_T_COEFF = "t_coefficient"
CONF_EVAL_TIME = "eval_time"
CONF_INITIAL_HVAC_MODE = "initial_hvac_mode"
CONF_PRECISION = "precision"
CONF_PLANIFICATEUR = "planificateur"
CONF_PRESENT = "present"
CONF_TEMPS_MIN = "temps_min"
CONF_REVERSE_ACTION = "reverse_action"

"""
CONF_MIN_TEMP = "min_temp"
CONF_MAX_TEMP = "max_temp"
CONF_AC_MODE = "ac_mode"
CONF_MIN_DUR = "min_cycle_duration"
CONF_COLD_TOLERANCE = "cold_tolerance"
CONF_HOT_TOLERANCE = "hot_tolerance"
CONF_KEEP_ALIVE = "keep_alive"
CONF_PRECISION = "precision"
CONF_TEMP_STEP = "target_temp_step"
"""

CONF_PRESETS = {
    p: f"{p}_temp"
    for p in (
        PRESET_AWAY,
        PRESET_COMFORT,
        PRESET_HOME,
        PRESET_SLEEP,
        PRESET_ACTIVITY,
    )
}

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_HEATER): cv.entity_id,
        vol.Required(CONF_IN_TEMP_SENSOR): cv.entity_id,
        vol.Required(CONF_OUT_TEMP_SENSOR): cv.entity_id,
        vol.Optional(CONF_T_COEFF, default=DEFAULT_T_COEFF): vol.Coerce(float),
        vol.Optional(CONF_C_COEFF, default=DEFAULT_C_COEFF): vol.Coerce(float),
        vol.Optional(CONF_EVAL_TIME): vol.Coerce(int),
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_TARGET_TEMP): vol.Coerce(float),
        vol.Optional(CONF_MIN_TEMP): vol.Coerce(float),
        vol.Optional(CONF_PLANIFICATEUR): vol.Coerce(list),
        vol.Optional(CONF_PRESENT, default=[]): vol.Coerce(list),
        vol.Optional(CONF_INITIAL_HVAC_MODE): vol.In(
            [HVACMode.AUTO, HVACMode.COOL, HVACMode.HEAT, HVACMode.OFF]
        ),
        vol.Optional(CONF_PRECISION): vol.In(
            [PRECISION_TENTHS, PRECISION_HALVES, PRECISION_WHOLE]
        ),
        vol.Optional(CONF_TEMPS_MIN, default=-1): vol.Coerce(int),
        vol.Optional(CONF_REVERSE_ACTION, default=True): vol.Coerce(boolean),
        vol.Optional(CONF_UNIQUE_ID): cv.string,
    }
).extend({vol.Optional(v): vol.Coerce(float) for (k, v) in CONF_PRESETS.items()})


async def async_setup_platform(
        hass: HomeAssistant,
        config: ConfigType,
        async_add_entities: AddEntitiesCallback,
        discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the generic thermostat platform."""

    await async_setup_reload_service(hass, DOMAIN, PLATFORMS)

    name = config.get(CONF_NAME)
    heater_entity_id = config.get(CONF_HEATER)
    in_temp_sensor_entity_id = config.get(CONF_IN_TEMP_SENSOR)
    out_temp_sensor_entity_id = config.get(CONF_OUT_TEMP_SENSOR)
    t_coeff = config.get(CONF_T_COEFF)
    c_coeff = config.get(CONF_C_COEFF)
    eval_time = config.get(CONF_EVAL_TIME)
    target_temp = config.get(CONF_TARGET_TEMP)
    min_temp = config.get(CONF_MIN_TEMP)
    planificateur = config.get(CONF_PLANIFICATEUR)
    present = config.get(CONF_PRESENT)
    temps_min = config.get(CONF_TEMPS_MIN)
    reverse_action = config.get(CONF_REVERSE_ACTION)
    initial_hvac_mode = config.get(CONF_INITIAL_HVAC_MODE)
    presets = {
        key: config[value] for key, value in CONF_PRESETS.items() if value in config
    }
    precision = config.get(CONF_PRECISION)
    unit = hass.config.units.temperature_unit
    unique_id = config.get(CONF_UNIQUE_ID)

    async_add_entities(
        [
            ThermostatTpi(
                name,
                heater_entity_id,
                in_temp_sensor_entity_id,
                out_temp_sensor_entity_id,
                t_coeff,
                c_coeff,
                eval_time,
                target_temp,
                min_temp,
                planificateur,
                present,
                temps_min,
                reverse_action,
                initial_hvac_mode,
                presets,
                precision,
                unit,
                unique_id,
            )
        ]
    )


class ThermostatTpi(ClimateEntity, RestoreEntity):
    """Representation of a Generic Thermostat device."""

    _attr_should_poll = False
    _attr_supported_features = (
            ClimateEntityFeature.TARGET_TEMPERATURE | ClimateEntityFeature.PRESET_MODE | ClimateEntityFeature.TURN_OFF | ClimateEntityFeature.TURN_ON
    )
    _enable_turn_on_off_backwards_compatibility = False

    def __init__(
            self,
            name,
            heater_entity_id,
            in_temp_sensor_entity_id,
            out_temp_sensor_entity_id,
            t_coeff,
            c_coeff,
            eval_time,
            target_temp,
            min_temp,
            planificateur,
            present,
            temps_min,
            reverse_action,
            initial_hvac_mode,
            presets,
            precision,
            unit,
            unique_id,
    ):
        """Initialize the thermostat."""
        self._attr_name = name
        self.heater_entity_id = heater_entity_id
        self.in_temp_sensor_entity_id = in_temp_sensor_entity_id
        self.out_temp_sensor_entity_id = out_temp_sensor_entity_id
        self.t_coeff = t_coeff
        self.c_coeff = c_coeff
        self.eval_time = eval_time
        self.ac_mode = False
        self._hvac_mode = initial_hvac_mode
        self._saved_target_temp = target_temp or next(iter(presets.values()), None)
        self._temp_precision = precision
        self._attr_hvac_modes = [HVACMode.AUTO, HVACMode.HEAT, HVACMode.COOL, HVACMode.OFF]
        self._active = False
        self._cur_in_temp = None
        self._cur_out_temp = None
        self._cur_power = 0
        self._temp_lock = asyncio.Lock()
        self._attr_preset_mode = PRESET_NONE
        self._target_temp = target_temp
        self._min_temp = min_temp
        self._planificateur = planificateur
        self._present = present
        self._temps_min = temps_min
        self._reverse_action = reverse_action
        self._attr_temperature_unit = unit
        self._attr_unique_id = unique_id
        self._last_on_time = None
        self._attr_supported_features = ClimateEntityFeature.TARGET_TEMPERATURE
        self._control_heating_off_call_later = None
        self._stop_control_loop = None
        if len(presets):
            self._attr_supported_features |= ClimateEntityFeature.PRESET_MODE
            self._attr_preset_modes = [PRESET_NONE] + list(presets.keys())
        else:
            self._attr_preset_modes = [PRESET_NONE]
        self._presets = presets

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added."""
        await super().async_added_to_hass()

        # Add listener
        self.async_on_remove(
            async_track_state_change_event(
                self.hass, [self.in_temp_sensor_entity_id, self.out_temp_sensor_entity_id], self._async_sensor_changed
            )
        )
        self.async_on_remove(
            async_track_state_change_event(
                self.hass, [self.heater_entity_id], self._async_switch_changed
            )
        )

        if self.eval_time:
            self._stop_control_loop = async_track_time_interval(
                self.hass,
                self._async_control_heating,
                timedelta(seconds=self.eval_time),
            )
            self.async_on_remove(self._stop_control_loop)

        @callback
        def _async_startup(*_):
            """Init on startup."""
            sensor_state = self.hass.states.get(self.in_temp_sensor_entity_id)
            if sensor_state and sensor_state.state not in (
                    STATE_UNAVAILABLE,
                    STATE_UNKNOWN,
            ):
                self._async_update_temp(
                    self.in_temp_sensor_entity_id, sensor_state
                )
                self.async_write_ha_state()

            out_temp_sensor_state = self.hass.states.get(self.out_temp_sensor_entity_id)
            if out_temp_sensor_state and out_temp_sensor_state.state not in (
                    STATE_UNAVAILABLE,
                    STATE_UNKNOWN,
            ):
                self._async_update_temp(
                    self.out_temp_sensor_entity_id, out_temp_sensor_state
                )
                self.async_write_ha_state()

            switch_state = self.hass.states.get(self.heater_entity_id)
            if switch_state and switch_state.state not in (
                    STATE_UNAVAILABLE,
                    STATE_UNKNOWN,
            ):
                self.hass.create_task(self._check_switch_initial_state())

        if self.hass.state == CoreState.running:
            _async_startup()
        else:
            self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_START, _async_startup)

        # Check If we have an old state
        if (old_state := await self.async_get_last_state()) is not None:
            # If we have no initial temperature, restore
            if self._target_temp is None:
                # If we have a previously saved temperature
                if old_state.attributes.get(ATTR_TEMPERATURE) is None:
                    if self.ac_mode:
                        self._target_temp = self.max_temp
                    else:
                        self._target_temp = self.min_temp
                    _LOGGER.warning(
                        "Undefined target temperature, falling back to %s",
                        self._target_temp,
                    )
                else:
                    self._target_temp = float(old_state.attributes[ATTR_TEMPERATURE])
            if (
                    self.preset_modes
                    and old_state.attributes.get(ATTR_PRESET_MODE) in self.preset_modes
            ):
                self._attr_preset_mode = old_state.attributes.get(ATTR_PRESET_MODE)
            if not self._hvac_mode and old_state.state:
                self._hvac_mode = old_state.state

        else:
            # No previous state, try and restore defaults
            if self._target_temp is None:
                self._target_temp = DEFAULT_TARGET_TEMPERATURE
            _LOGGER.warning(
                "No previously saved temperature, setting to %s", self._target_temp
            )

        # Set default state to off
        if not self._hvac_mode:
            self._hvac_mode = HVACMode.OFF

        await self._async_control_heating()

    @property
    def precision(self):
        """Return the precision of the system."""
        if self._temp_precision is not None:
            return self._temp_precision
        return super().precision

    @property
    def target_temperature_step(self):
        """Return the supported step of target temperature."""
        # Since this integration does not yet have a step size parameter
        # we have to re-use the precision as the step size for now.
        return self.precision

    @property
    def current_temperature(self):
        """Return the sensor temperature."""
        return self._cur_in_temp

    @property
    def hvac_mode(self):
        """Return current operation."""
        return self._hvac_mode

    @property
    def hvac_action(self):
        """Return the current running hvac operation if supported.

        Need to be one of CURRENT_HVAC_*.
        """
        if self._hvac_mode == HVACMode.OFF:
            return HVACAction.OFF
        if not self._is_device_active:
            return HVACAction.IDLE if self._reverse_action else HVACAction.HEATING
        if self.ac_mode:
            return HVACAction.COOLING
        return HVACAction.HEATING if self._reverse_action else HVACAction.IDLE

    @property
    def target_temperature(self):
        """Return the temperature we try to reach."""
        return self._target_temp

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        self._hvac_mode = hvac_mode
        if hvac_mode == HVACMode.OFF:
            if self._stop_control_loop:
                self._stop_control_loop()
                if self._control_heating_off_call_later:
                    self._control_heating_off_call_later()
            await self._async_heater_turn_off()
        else:
            # await self._async_control_heating()
            await self._update_control_loop()
        # Ensure we update the current operation after changing the mode
        self.async_write_ha_state()

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set new target temperature."""
        if (temperature := kwargs.get(ATTR_TEMPERATURE)) is None:
            return
        self._target_temp = temperature
        await self._async_control_heating()
        self.async_write_ha_state()

    async def _update_control_loop(self):
        """Reset control loop to give its latest state."""
        if self._stop_control_loop:
            self._stop_control_loop()  # Call remove callback on the loop
            if self._control_heating_off_call_later:
                self._control_heating_off_call_later()

        # Update heating status
        await self._async_control_heating()

        # Start a new control loop with updated values
        self._stop_control_loop = async_track_time_interval(
            self.hass, self._async_control_heating, timedelta(seconds=self.eval_time)
        )
        self.async_on_remove(self._stop_control_loop)

    async def _async_temp_sensor_changed(self, event):
        """Handle temperature changes."""
        new_state = event.data.get("new_state")
        entity_id = event.data.get("entity_id")
        if new_state is None or new_state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            return

        self._async_update_temp(entity_id, new_state)
        self.async_write_ha_state()

    async def _async_sensor_changed(self, event):
        """Handle temperature changes."""
        new_state = event.data.get("new_state")
        entity_id = event.data.get("entity_id")
        if new_state is None or new_state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            return

        self._async_update_temp(entity_id, new_state)
        await self._async_control_heating()
        self.async_write_ha_state()

    async def _check_switch_initial_state(self):
        """Prevent the device from keep running if HVACMode.OFF."""
        if self._hvac_mode == HVACMode.OFF and self._is_device_active:
            _LOGGER.warning(
                "The climate mode is OFF, but the switch device is ON. Turning off device %s",
                self.heater_entity_id,
            )
            await self._async_heater_turn_off()

    @callback
    def _async_switch_changed(self, event):
        """Handle heater switch state changes."""
        new_state = event.data.get("new_state")
        old_state = event.data.get("old_state")
        if new_state is None:
            return
        if old_state is None:
            self.hass.create_task(self._check_switch_initial_state())
        self.async_write_ha_state()

    @callback
    def _async_update_temp(self, entity_id, state):
        try:
            cur_temp = float(state.state)
            if math.isnan(cur_temp) or math.isinf(cur_temp):
                raise ValueError(f"Sensor has illegal state {state.state}")
            if entity_id == self.in_temp_sensor_entity_id:
                self._cur_in_temp = cur_temp
            elif entity_id == self.out_temp_sensor_entity_id:
                self._cur_out_temp = cur_temp
            else:
                _LOGGER.error("Unable to update from sensor: no matching entity id")
        except ValueError as ex:
            _LOGGER.error("Unable to update from sensor: %s", ex)

    @callback
    def _async_update_power(self):
        """Update power with latest state from sensors."""
        try:
            c = self.c_coeff
            t = self.t_coeff
            target = self.target_temperature if self._hvac_mode == HVACMode.HEAT or (
                        self._hvac_mode == HVACMode.AUTO and self._planificateur_in_periode()) else self._min_temp
            inside = self._cur_in_temp
            outside = self._cur_out_temp
            if None in [inside, outside, target]:
                return 0
            power_formula = c * (target - inside) + t * (target - outside)
            cur_power = min(max(power_formula, 0), 1) * 100

            self._cur_power = cur_power
        except ValueError as ex:
            _LOGGER.error("Unable to compute power from sensor: %s", ex)

    async def _async_control_heating(self, time=None):
        """Check if we need to turn heating on or off."""
        async with self._temp_lock:
            if not self._active and None not in (
                    self._cur_in_temp,
                    self._cur_out_temp,
                    self._target_temp,
            ):
                self._active = True
                _LOGGER.info(
                    "Obtained current and target temperature. "
                    "Generic thermostat active. %s, %s",
                    self._cur_in_temp,
                    self._target_temp,
                )

            """self._window_opened or gestion de la fenetre """
            if self._hvac_mode == HVACMode.OFF:
                return

            self._async_update_power()
            _LOGGER.info("Current power %s", self._cur_power)

            if not self._cur_power:
                await self._async_heater_turn_off()
                return

            # Vérification du temps minimum entre deux lancements
            if self._temps_min > 1 and self._last_on_time is not None:
                diff = (datetime.now() - self._last_on_time).total_seconds()
                if diff < self._temps_min:
                    _LOGGER.info("Attente de %s secondes avant relance (écoulé: %s)", self._temps_min, round(diff))
                    return

            heating_delay = self._cur_power * round(self.eval_time / 100, 2)

            if self._temps_min > -1 and heating_delay < self._temps_min:
                heating_delay = self._temps_min

            _LOGGER.info("Turning on heater %s", self.heater_entity_id)
            ## forcer le switch
            await self._async_heater_turn_on()
            self._last_on_time = datetime.now()

            _LOGGER.info("Waiting for %s before turning it down", heating_delay)
            # Stop callback if it was already existing
            if self._control_heating_off_call_later:
                self._control_heating_off_call_later()
            self._control_heating_off_call_later = async_call_later(
                self.hass, heating_delay, self._async_control_heating_off_cb
            )

            self.async_on_remove(self._control_heating_off_call_later)

    async def _async_control_heating_off_cb(self):
        """Callback called after heating time to stop heating."""
        _LOGGER.info("Turning heater to eco mode to cool off %s", self.heater_entity_id)
        await self._async_heater_turn_off()

    """" @todo: a regarder """

    @property
    def _is_device_active(self):
        """If the toggleable device is currently active."""
        if not self.hass.states.get(self.heater_entity_id):
            return None

        return self.hass.states.is_state(self.heater_entity_id, STATE_OFF)

    async def _async_heater_turn_on(self):
        """Turn heater toggleable device on."""
        data = {ATTR_ENTITY_ID: self.heater_entity_id}
        action = SERVICE_TURN_OFF if self._reverse_action else SERVICE_TURN_ON
        await self.hass.services.async_call(
            HA_DOMAIN, action, data, context=self._context
        )

    async def _async_heater_turn_off(self):
        """Turn heater toggleable device off."""
        data = {ATTR_ENTITY_ID: self.heater_entity_id}
        action = SERVICE_TURN_ON if self._reverse_action else SERVICE_TURN_OFF
        await self.hass.services.async_call(
            HA_DOMAIN, action, data, context=self._context
        )

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set new preset mode."""
        if preset_mode not in (self.preset_modes or []):
            raise ValueError(
                f"Got unsupported preset_mode {preset_mode}. Must be one of {self.preset_modes}"
            )
        if preset_mode == self._attr_preset_mode:
            # I don't think we need to call async_write_ha_state if we didn't change the state
            return
        if preset_mode == PRESET_NONE:
            self._attr_preset_mode = PRESET_NONE
            self._target_temp = self._saved_target_temp
            await self._async_control_heating()
        else:
            if self._attr_preset_mode == PRESET_NONE:
                self._saved_target_temp = self._target_temp
            self._attr_preset_mode = preset_mode
            self._target_temp = self._presets[preset_mode]
            await self._async_control_heating()

        self.async_write_ha_state()

    def _planificateur_in_periode(self) -> bool:
        periode = False
        day = datetime.now().weekday()
        if self._planificateur[day] != "none" and self._is_present():
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            plages = self._planificateur[day].split(",")
            i = 0
            while i < len(plages) and not periode:
                plage = plages[i].split("-")
                if plage[0] <= current_time <= plage[1]:
                    periode = True
                i += 1

        return periode

    def _is_present(self) -> bool:
        present = False
        i = 0
        while i < len(self._present) and not present:
            if self.hass.states.is_state(self._present[i], STATE_HOME):
                present = True
            i += 1
        return present or len(self._present) == 0
