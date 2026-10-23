from __future__ import annotations

import threading

import pytest

from flea import CameraProperty, CameraPropertyInfo, CameraPropertyState, PropertyType
from flea.controls import TerminalControls, resolve_property


def property_state(
    property_type: PropertyType,
    *,
    value: float = 5.0,
    auto: bool = False,
    absolute: bool = True,
    one_push: bool = False,
) -> CameraPropertyState:
    info = CameraPropertyInfo(
        type=property_type,
        present=True,
        auto_supported=True,
        manual_supported=True,
        on_off_supported=True,
        one_push_supported=one_push,
        absolute_supported=absolute,
        readable=True,
        minimum=0,
        maximum=1_000,
        absolute_minimum=0.01,
        absolute_maximum=100.0,
        units="milliseconds",
        unit_abbreviation="ms",
    )
    current = CameraProperty(
        type=property_type,
        present=True,
        absolute=absolute,
        one_push=False,
        auto=auto,
        enabled=True,
        value_a=int(value),
        value_b=20,
        value=value,
    )
    return CameraPropertyState(info=info, value=current)


class FakeSession:
    def __init__(self) -> None:
        self.states = {
            property_type: property_state(property_type)
            for property_type in PropertyType
        }
        self.writes: list[tuple[PropertyType, float | int, dict[str, object]]] = []

    def list_properties(self):
        return self.states

    def set_property_sync(
        self, property_type: PropertyType, value: float | int, **options: object
    ) -> None:
        self.writes.append((property_type, value, options))
        state = self.states[property_type]
        current = state.value
        assert current is not None
        self.states[property_type] = CameraPropertyState(
            info=state.info,
            value=CameraProperty(
                type=property_type,
                present=True,
                absolute=bool(options["absolute"]),
                one_push=bool(options.get("one_push")),
                auto=bool(options["auto"]),
                enabled=bool(options["enabled"]),
                value_a=int(value),
                value_b=int(options.get("value_b") or 0),
                value=float(value),
            ),
        )


def test_resolve_property_accepts_aliases() -> None:
    assert resolve_property("shutter") is PropertyType.SHUTTER
    assert resolve_property("exposure-time") is PropertyType.SHUTTER
    assert resolve_property("fps") is PropertyType.FRAME_RATE


def test_terminal_set_and_auto_commands() -> None:
    session = FakeSession()
    controls = TerminalControls(session, threading.Event())
    result = controls.execute("set shutter 12.5")
    assert "12.5 ms" in result[0]
    assert session.writes[-1][0:2] == (PropertyType.SHUTTER, 12.5)
    controls.execute("auto shutter on")
    assert session.writes[-1][2]["auto"] is True


def test_terminal_integer_pair_and_quit_commands() -> None:
    session = FakeSession()
    state = property_state(PropertyType.WHITE_BALANCE, absolute=False)
    session.states[PropertyType.WHITE_BALANCE] = state
    stop = threading.Event()
    controls = TerminalControls(session, stop)
    controls.execute("set white_balance 600 800")
    assert session.writes[-1][0:2] == (PropertyType.WHITE_BALANCE, 600)
    assert session.writes[-1][2]["value_b"] == 800
    assert controls.execute("quit") == ["stopping preview"]
    assert stop.is_set()


def test_terminal_distinguishes_absolute_and_raw_values() -> None:
    session = FakeSession()
    controls = TerminalControls(session, threading.Event())

    for invalid in ("set gamma 0", "set gamma 1332"):
        with pytest.raises(ValueError, match="absolute mode"):
            controls.execute(invalid)

    result = controls.execute("setraw gamma 900")
    assert "raw=900" in result[0]
    assert "abs-range=" in result[0]
    assert "raw-range=" in result[0]
    assert session.writes[-1][1] == 900
    assert session.writes[-1][2]["absolute"] is False
