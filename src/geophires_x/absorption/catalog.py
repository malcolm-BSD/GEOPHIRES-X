"""Catalog utilities for commercial absorption chillers.

Supports an embedded default CSV that ships with the package, user CSV
overrides, and a stub for remote queries. CatalogEntry is a thin container
for each row.
"""
from typing import Any, Dict, List, Optional
import csv
import pkgutil
import io


class CatalogEntry:
    """Container for a single catalog record. Fields mirror CSV columns."""

    def __init__(self, **kwargs) -> None:
        self.data: Dict[str, Any] = kwargs

    def __getitem__(self, key: str) -> Any:
        return self.data.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


class Catalog:
    """Maintain an embedded default dataset, allow CSV import, and remote query.

    The embedded CSV should be stored under the `data/` directory in the
    package. If no embedded CSV is found the catalog is empty until a user CSV
    or remote data is loaded.
    """

    def __init__(self, embedded_csv_path: Optional[str] = None) -> None:
        self.entries: List[CatalogEntry] = []
        self.embedded_path = embedded_csv_path or "data/absorption_chiller_catalog_default.csv"
        self._load_embedded()

    def _load_embedded(self) -> None:
        try:
            data = pkgutil.get_data(__package__.split(".")[0], self.embedded_path)
            if data:
                text = data.decode("utf-8")
                reader = csv.DictReader(io.StringIO(text))
                for row in reader:
                    self.entries.append(CatalogEntry(**row))
        except Exception:
            # Embedded data not found or failed to load; catalog remains empty
            pass

    def load_user_csv(self, csv_path: str) -> None:
        """Load user-provided CSV and append/override embedded entries."""
        try:
            with open(csv_path, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    self.entries.append(CatalogEntry(**row))
        except Exception:
            raise

    def query(self, capacity_kW: float, refrigerant_family: Optional[str] = None, effect_type: Optional[str] = None) -> List[CatalogEntry]:
        """Return candidate entries matching constraints.

        This is a simple filter by nominal capacity and optional string matches.
        """
        candidates: List[CatalogEntry] = []
        for e in self.entries:
            try:
                cap = float(e.get("nominal_cooling_kW", 0))
            except Exception:
                continue
            if cap < 0.5 * capacity_kW:
                # skip units that are too small to be practical
                continue
            if refrigerant_family and e.get("refrigerant_family") != refrigerant_family:
                continue
            if effect_type and e.get("effect_type") != effect_type:
                continue
            candidates.append(e)
        return candidates

    def select_min_cost_set(self, required_capacity_kW: float, allow_staging: bool = True) -> Dict[str, Any]:
        """Return a selected set of units and estimated cost.

        This is a naive greedy packer by descending nominal size that tries to
        reach the required capacity.
        """
        sorted_entries = sorted(self.entries, key=lambda e: float(e.get("nominal_cooling_kW", 0)), reverse=True)
        remaining = required_capacity_kW
        selection: List[Dict[str, Any]] = []
        total_capacity = 0.0
        total_cost = 0.0
        for e in sorted_entries:
            try:
                cap = float(e.get("nominal_cooling_kW", 0))
            except Exception:
                continue
            if cap <= 0:
                continue
            count = int(remaining // cap)
            if remaining % cap > 0:
                count += 1
            if count <= 0:
                continue
            selection.append({"model_id": e.get("model_id"), "count": count, "nominal_kW": cap})
            total_capacity += cap * count
            try:
                total_cost += float(e.get("installed_cost_USD", 0)) * count
            except Exception:
                total_cost += 0.0
            remaining = max(0.0, remaining - cap * count)
            if remaining <= 0:
                break
        return {"selected": selection, "total_capacity_kW": total_capacity, "estimated_cost_USD": total_cost}

    def query_remote_catalog(self, query_params: Dict[str, Any], timeout_s: int = 10) -> List[CatalogEntry]:
        """Best-effort remote query. Returns empty list by default.

        Implementers can override this method to call real endpoints and cache
        results locally.
        """
        return []

