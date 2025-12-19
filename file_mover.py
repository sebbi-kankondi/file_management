"""
Automated file management utility.

This module loads configuration describing how to move files from source
directories (e.g., Downloads) into destination folders based on filename
patterns and extensions. The implementation follows the operational layout
described in ``operational_layout.txt``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import fnmatch
import re
import time

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


@dataclass(frozen=True)
class Rule:
    """Represents a classification rule."""

    match: str
    values: List[str]
    dest: Path

    def normalized_values(self) -> List[str]:
        return [value.lower() for value in self.values]


@dataclass
class MoveOptions:
    """Options that control runtime behavior."""

    skip_if_locked: bool = True
    min_age_minutes: int = 2
    on_conflict: str = "append_timestamp"
    dry_run: bool = False
    log_file: Path = Path("~/.local/state/file_mover.log").expanduser()
    ignore_patterns: List[str] = field(default_factory=lambda: [".part", ".crdownload", "~$", ".tmp"])
    year_month_subfolders: bool = False
    quarantine_dir: Path = Path("~/Downloads/Quarantine").expanduser()
    verify_checksum: bool = True


@dataclass
class FileMoverConfig:
    """Configuration container for the file mover."""

    sources: List[Path]
    rules: List[Rule]
    options: MoveOptions

    @classmethod
    def from_mapping(cls, data: dict) -> "FileMoverConfig":
        sources = [Path(path).expanduser() for path in data.get("sources", ["~/Downloads"])]

        raw_rules = data.get("rules", [])
        rules: List[Rule] = []
        for rule_data in raw_rules:
            match = rule_data.get("match")
            values = rule_data.get("values", [])
            dest = rule_data.get("dest")

            if not match or dest is None:
                raise ValueError(f"Rule missing required fields: {rule_data}")

            rules.append(
                Rule(
                    match=str(match),
                    values=[str(value) for value in values],
                    dest=Path(dest).expanduser(),
                )
            )

        options_data = data.get("options", {})
        options = MoveOptions(
            skip_if_locked=bool(options_data.get("skip_if_locked", True)),
            min_age_minutes=int(options_data.get("min_age_minutes", 2)),
            on_conflict=str(options_data.get("on_conflict", "append_timestamp")),
            dry_run=bool(options_data.get("dry_run", False)),
            log_file=Path(options_data.get("log_file", "~/.local/state/file_mover.log")).expanduser(),
            ignore_patterns=[str(pattern) for pattern in options_data.get("ignore_patterns", MoveOptions().ignore_patterns)],
            year_month_subfolders=bool(options_data.get("year_month_subfolders", False)),
            quarantine_dir=Path(options_data.get("quarantine_dir", MoveOptions().quarantine_dir)).expanduser(),
            verify_checksum=bool(options_data.get("verify_checksum", True)),
        )

        return cls(sources=sources, rules=rules, options=options)


def _load_yaml(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load YAML configuration files.")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(config_path: Optional[Path]) -> FileMoverConfig:
    """
    Load configuration from a JSON or YAML file.

    If ``config_path`` is None, default values are used.
    """
    if config_path is None:
        logging.getLogger(__name__).info("No config path provided; using defaults.")
        return FileMoverConfig.from_mapping({})

    config_path = config_path.expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix in {".json"}:
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    elif suffix in {".yml", ".yaml"}:
        data = _load_yaml(config_path)
    else:
        raise ValueError(f"Unsupported configuration format: {suffix}")

    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping, got {type(data).__name__}")

    return FileMoverConfig.from_mapping(data)


@dataclass(frozen=True)
class ClassificationResult:
    """Details about how a file was classified."""

    rule: Rule
    destination: Path


@dataclass
class MoveRecord:
    source: Path
    destination: Path
    rule: Rule
    action: str
    error: Optional[str] = None
    duration_seconds: float = 0.0


def _file_age_minutes(path: Path) -> float:
    stat_result = path.stat()
    age_seconds = time.time() - stat_result.st_mtime
    return age_seconds / 60


def _is_locked(path: Path) -> bool:
    try:
        with path.open("a"):
            return False
    except OSError:
        return True


def should_ignore(path: Path, ignore_patterns: Iterable[str]) -> bool:
    name = path.name
    for pattern in ignore_patterns:
        if fnmatch.fnmatch(name, pattern):
            return True
    return False


def discover_files(config: FileMoverConfig) -> List[Path]:
    """Return files from all sources that are eligible for processing."""

    candidates: List[Path] = []
    for source in config.sources:
        if not source.exists():
            continue
        for item in source.iterdir():
            if not item.is_file():
                continue
            if should_ignore(item, config.options.ignore_patterns):
                continue
            if config.options.skip_if_locked and _is_locked(item):
                continue
            if _file_age_minutes(item) < config.options.min_age_minutes:
                continue
            candidates.append(item)
    return candidates


class RuleMatcher:
    """Applies classification rules to filesystem entries."""

    def __init__(self, rules: Iterable[Rule], default_unsorted: Optional[Path] = None):
        self.rules = list(rules)
        self.default_unsorted = default_unsorted or Path("~/Downloads/Unsorted").expanduser()

    def match(self, path: Path) -> ClassificationResult:
        name = path.name.lower()
        suffix = path.suffix.lower().lstrip(".")

        for rule in self.rules:
            if rule.match == "extension":
                if suffix in rule.normalized_values():
                    return ClassificationResult(rule=rule, destination=rule.dest)
            elif rule.match == "name_contains":
                if any(value in name for value in rule.normalized_values()):
                    return ClassificationResult(rule=rule, destination=rule.dest)
            elif rule.match == "regex":
                for pattern in rule.values:
                    if re.search(pattern, name):
                        return ClassificationResult(rule=rule, destination=rule.dest)
            elif rule.match == "any":
                return ClassificationResult(rule=rule, destination=rule.dest)

        fallback_rule = Rule(match="any", values=["*"], dest=self.default_unsorted)
        return ClassificationResult(rule=fallback_rule, destination=self.default_unsorted)


class FileMover:
    """Coordinates discovery, classification, and movement of files."""

    def __init__(self, config: FileMoverConfig):
        self.config = config
        self.rule_matcher = RuleMatcher(config.rules)
        self.logger = self._configure_logger(config.options.log_file)

    def _configure_logger(self, log_path: Path) -> logging.Logger:
        logger = logging.getLogger("file_mover")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_path)
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _year_month_destination(self, dest: Path) -> Path:
        if not self.config.options.year_month_subfolders:
            return dest
        now = datetime.now()
        return dest / f"{now.year:04d}" / f"{now.month:02d}"

    def _resolve_conflict(self, dest: Path) -> Path:
        if not dest.exists():
            return dest

        strategy = self.config.options.on_conflict
        stem = dest.stem
        suffix = dest.suffix
        parent = dest.parent

        if strategy == "append_timestamp":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return parent / f"{stem}_{timestamp}{suffix}"

        counter = 1
        while True:
            candidate = parent / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    def _checksum(self, path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _move_file(self, src: Path, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)

        src_dev = src.stat().st_dev
        dest_dev = dest.parent.stat().st_dev
        if src_dev == dest_dev:
            shutil.move(str(src), str(dest))
            return

        temp_dest = dest.with_suffix(dest.suffix + ".tmp")
        shutil.copy2(src, temp_dest)

        if self.config.options.verify_checksum:
            if self._checksum(src) != self._checksum(temp_dest):
                temp_dest.unlink(missing_ok=True)
                raise RuntimeError(f"Checksum mismatch when copying {src} to {dest}")

        shutil.move(str(temp_dest), str(dest))
        src.unlink()

    def process(self, files: Optional[Iterable[Path]] = None) -> List[MoveRecord]:
        files_to_process = list(files) if files is not None else discover_files(self.config)
        results: List[MoveRecord] = []

        for file_path in files_to_process:
            start = time.time()
            try:
                classification = self.rule_matcher.match(file_path)
                destination_dir = self._year_month_destination(classification.destination)
                final_destination = self._resolve_conflict(destination_dir / file_path.name)

                if self.config.options.dry_run:
                    action = "dry-run"
                else:
                    self._move_file(file_path, final_destination)
                    action = "moved"

                duration = time.time() - start
                record = MoveRecord(
                    source=file_path,
                    destination=final_destination,
                    rule=classification.rule,
                    action=action,
                    duration_seconds=duration,
                )
                self.logger.info(
                    "action=%s source=%s destination=%s rule=%s duration=%.3fs",
                    action,
                    file_path,
                    final_destination,
                    classification.rule.match,
                    duration,
                )
                results.append(record)
            except Exception as exc:  # pragma: no cover - runtime safeguard
                duration = time.time() - start
                quarantine_target = self.config.options.quarantine_dir / file_path.name
                quarantine_target.parent.mkdir(parents=True, exist_ok=True)
                if not self.config.options.dry_run and file_path.exists():
                    shutil.move(str(file_path), str(quarantine_target))
                error_message = str(exc)
                self.logger.error(
                    "action=error source=%s destination=%s error=%s duration=%.3fs",
                    file_path,
                    quarantine_target,
                    error_message,
                    duration,
                )
                results.append(
                    MoveRecord(
                        source=file_path,
                        destination=quarantine_target,
                        rule=Rule(match="error", values=[], dest=quarantine_target),
                        action="error",
                        error=error_message,
                        duration_seconds=duration,
                    )
                )
        return results


def _apply_overrides(config: FileMoverConfig, args: argparse.Namespace) -> FileMoverConfig:
    if args.source:
        config.sources = [Path(path).expanduser() for path in args.source]
    if args.dry_run:
        config.options.dry_run = True
    if args.min_age is not None:
        config.options.min_age_minutes = int(args.min_age)
    return config


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatically relocate files from Downloads into organized folders.")
    parser.add_argument("--config", type=Path, help="Path to YAML/JSON configuration file.")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without moving files.")
    parser.add_argument("--source", action="append", help="Override sources (can be provided multiple times).")
    parser.add_argument("--min-age", type=int, help="Minimum age in minutes before a file is moved.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)
    config = _apply_overrides(config, args)

    mover = FileMover(config)
    results = mover.process()

    moved = sum(1 for result in results if result.action == "moved")
    skipped = sum(1 for result in results if result.action == "dry-run")
    errors = [result for result in results if result.action == "error"]

    print(f"Moved: {moved}, Dry-run: {skipped}, Errors: {len(errors)}")
    if errors:
        print("Errors encountered:")
        for error in errors:
            print(f" - {error.source} -> {error.destination}: {error.error}")

    return 1 if errors else 0


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    sys.exit(main())
