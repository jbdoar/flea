from __future__ import annotations

import shlex
import threading
from collections.abc import Callable

from .session import CaptureSession
from .types import CameraProperty, CameraPropertyState, PropertyType

_PROPERTY_NAMES = {property_type.name.lower(): property_type for property_type in PropertyType}
_PROPERTY_NAMES.update(
    {
        "exposure_time": PropertyType.SHUTTER,
        "exposure-time": PropertyType.SHUTTER,
        "fps": PropertyType.FRAME_RATE,
    }
)


def resolve_property(name: str) -> PropertyType:
    normalized = name.strip().lower().replace("-", "_")
    try:
        return _PROPERTY_NAMES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown camera property: {name}") from exc


def _current_value(value: CameraProperty) -> float | int:
    return value.value if value.absolute else value.value_a


def describe_property(property_type: PropertyType, state: CameraPropertyState) -> str:
    info = state.info
    name = property_type.name.lower()
    if not info.present:
        return f"{name:<18} unavailable"
    if state.value is None:
        current = "write-only"
    elif state.value.absolute:
        unit = info.unit_abbreviation or info.units
        current = f"{state.value.value:.6g} {unit}".rstrip()
    elif property_type is PropertyType.WHITE_BALANCE:
        current = f"red={state.value.value_a} blue={state.value.value_b}"
    else:
        current = str(state.value.value_a)
    modes = []
    if state.value is not None:
        modes.append("auto" if state.value.auto else "manual")
        modes.append("on" if state.value.enabled else "off")
    if info.absolute_supported:
        unit = info.unit_abbreviation or info.units
        limits = f"range={info.absolute_minimum:g}..{info.absolute_maximum:g} {unit}".rstrip()
    else:
        limits = f"range={info.minimum}..{info.maximum}"
    capabilities = []
    if info.auto_supported:
        capabilities.append("auto")
    if info.one_push_supported:
        capabilities.append("onepush")
    suffix = f" capabilities={','.join(capabilities)}" if capabilities else ""
    return f"{name:<18} {current:<24} {'/'.join(modes):<11} {limits}{suffix}"


def describe_properties(states: dict[PropertyType, CameraPropertyState]) -> list[str]:
    return [
        describe_property(property_type, states[property_type])
        for property_type in PropertyType
    ]


class TerminalControls:
    """Small interactive command shell for a running capture session."""

    def __init__(self, session: CaptureSession, stop_event: threading.Event) -> None:
        self.session = session
        self.stop_event = stop_event

    @staticmethod
    def help_lines() -> list[str]:
        return [
            "Camera controls:",
            "  list",
            "  get PROPERTY",
            "  set PROPERTY VALUE [VALUE_B]",
            "  auto PROPERTY on|off",
            "  enable PROPERTY on|off",
            "  onepush PROPERTY",
            "  help",
            "  quit",
            "Examples: set shutter 5 | set gain 3 | set white_balance 600 800",
        ]

    def run(
        self,
        *,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        for line in self.help_lines():
            output_fn(line)
        while not self.stop_event.is_set():
            try:
                command = input_fn("flea> ")
            except (EOFError, KeyboardInterrupt):
                self.stop_event.set()
                return
            if not command.strip():
                continue
            try:
                for line in self.execute(command):
                    output_fn(line)
            except Exception as exc:
                output_fn(f"error: {exc}")

    def execute(self, command: str) -> list[str]:
        parts = shlex.split(command)
        if not parts:
            return []
        action = parts[0].lower()
        if action in ("quit", "exit"):
            self.stop_event.set()
            return ["stopping preview"]
        if action == "help":
            return self.help_lines()
        if action == "list":
            return describe_properties(self.session.list_properties())
        if action == "get" and len(parts) == 2:
            property_type = resolve_property(parts[1])
            state = self.session.list_properties()[property_type]
            return [describe_property(property_type, state)]
        if action == "set" and len(parts) in (3, 4):
            property_type = resolve_property(parts[1])
            state = self.session.list_properties()[property_type]
            self._require_manual(state)
            value_b = int(parts[3]) if len(parts) == 4 else None
            absolute = state.info.absolute_supported and value_b is None
            value: float | int = float(parts[2]) if absolute else int(parts[2])
            self.session.set_property_sync(
                property_type,
                value,
                value_b=value_b,
                absolute=absolute,
                auto=False,
                enabled=True,
            )
            return self.execute(f"get {parts[1]}")
        if action in ("auto", "enable") and len(parts) == 3:
            property_type = resolve_property(parts[1])
            enabled = self._parse_switch(parts[2])
            state = self.session.list_properties()[property_type]
            if state.value is None:
                raise ValueError(f"{parts[1]} is not readable")
            if action == "auto" and not state.info.auto_supported:
                raise ValueError(f"{parts[1]} does not support automatic mode")
            if action == "enable" and not state.info.on_off_supported:
                raise ValueError(f"{parts[1]} does not support on/off control")
            self.session.set_property_sync(
                property_type,
                _current_value(state.value),
                value_b=state.value.value_b,
                absolute=state.value.absolute,
                auto=enabled if action == "auto" else state.value.auto,
                enabled=enabled if action == "enable" else state.value.enabled,
            )
            return self.execute(f"get {parts[1]}")
        if action == "onepush" and len(parts) == 2:
            property_type = resolve_property(parts[1])
            state = self.session.list_properties()[property_type]
            if state.value is None or not state.info.one_push_supported:
                raise ValueError(f"{parts[1]} does not support one-push adjustment")
            self.session.set_property_sync(
                property_type,
                _current_value(state.value),
                value_b=state.value.value_b,
                absolute=state.value.absolute,
                auto=state.value.auto,
                enabled=state.value.enabled,
                one_push=True,
            )
            return [f"started one-push adjustment for {property_type.name.lower()}"]
        raise ValueError("Invalid command; enter 'help' for syntax")

    @staticmethod
    def _require_manual(state: CameraPropertyState) -> None:
        if not state.info.present:
            raise ValueError("Property is unavailable on this camera")
        if not state.info.manual_supported:
            raise ValueError("Property is not manually writable")

    @staticmethod
    def _parse_switch(value: str) -> bool:
        normalized = value.lower()
        if normalized in ("on", "true", "1", "yes"):
            return True
        if normalized in ("off", "false", "0", "no"):
            return False
        raise ValueError("Expected on or off")
