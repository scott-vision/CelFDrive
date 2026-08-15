"""Project-level ordered phase configuration for CellClicker."""

import json
import os
import hashlib


DEFAULT_PHASES = (
    "prophase",
    "earlyprometaphase",
    "prometaphase",
    "metaphase",
    "anaphase",
    "telophase",
)
SETTINGS_FILENAME = "phase_selector_settings.json"


def settings_path(project_dir):
    return os.path.join(project_dir, "user_selections", SETTINGS_FILENAME)


def load_phases(project_dir):
    """Return configured phase names, or the mitosis preset for legacy projects."""
    try:
        with open(settings_path(project_dir), encoding="utf-8") as handle:
            entries = json.load(handle)["phases"]
        return validate_phase_entries(entries)
    except FileNotFoundError:
        return list(DEFAULT_PHASES)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid phase settings in `{settings_path(project_dir)}`: {exc}") from exc


def validate_phase_entries(entries):
    """Validate ``[{id, name}]`` settings and return names in numeric order."""
    normalized = []
    for entry in entries:
        try:
            identifier = int(entry["id"])
            name = str(entry["name"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Each phase needs an integer ID and non-empty name.") from exc
        if not name or not name.replace("_", "").isalnum():
            raise ValueError(f"Phase name `{name}` must use letters, numbers, and underscores only.")
        normalized.append((identifier, name))
    normalized.sort()
    if not normalized:
        raise ValueError("Configure at least one phase.")
    if [identifier for identifier, _ in normalized] != list(range(len(normalized))):
        raise ValueError("Phase IDs must be unique and consecutive, starting at 0.")
    names = [name for _, name in normalized]
    if len(set(names)) != len(names):
        raise ValueError("Phase names must be unique.")
    return names


def save_phases(project_dir, entries):
    """Save a validated ordered phase mapping for one project."""
    names = validate_phase_entries(entries)
    os.makedirs(os.path.dirname(settings_path(project_dir)), exist_ok=True)
    with open(settings_path(project_dir), "w", encoding="utf-8") as handle:
        json.dump(
            {"phases": [{"id": index, "name": name} for index, name in enumerate(names)]},
            handle, indent=2,
        )
    return names


def phase_signature(phases):
    """Return the stable signature that invalidates selections after remapping."""
    return hashlib.sha256("\n".join(phases).encode("utf-8")).hexdigest()
