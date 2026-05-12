"""Catalog utilities for commercial absorption chillers.

The catalog supports the embedded GEOPHIRES seed CSV, user-provided CSV
overrides, and best-effort remote JSON/CSV endpoints. Embedded rows are
engineering-estimate seed data, not procurement-grade vendor quotes; provenance
columns are preserved so callers can decide whether a row is sufficiently
verified for their use case.
"""

import csv
import io
import json
import pkgutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROVENANCE_COLUMNS = ("source", "source_url", "last_verified", "license_note")


class CatalogEntry:
    """Container for a single catalog record.

    Parameters
    ----------
    kwargs:
        Catalog row fields. Keys mirror CSV columns and are intentionally kept
        flexible so user and remote catalogs can add extra metadata without a
        schema migration.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.data: Dict[str, Any] = kwargs

    def __getitem__(self, key: str) -> Any:
        """Return the value for ``key`` or ``None`` when absent."""
        return self.data.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for ``key`` or ``default`` when absent."""
        return self.data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Return a shallow dictionary copy of the catalog row."""
        return dict(self.data)


class Catalog:
    """Maintain an embedded default dataset, allow CSV import, and remote query.

    Parameters
    ----------
    embedded_csv_path:
        Optional path to the embedded/default CSV. Relative paths are resolved
        against package resources and then the repository root.
    remote_catalog_url:
        Optional default endpoint used by :meth:`query_remote_catalog`.
    cache_path:
        Optional JSON cache file used for successful remote query results.
    """

    def __init__(
        self,
        embedded_csv_path: Optional[str] = None,
        remote_catalog_url: Optional[str] = None,
        cache_path: Optional[str] = None,
    ) -> None:
        self.entries: List[CatalogEntry] = []
        self.embedded_path = embedded_csv_path or "data/absorption_chiller_catalog_default.csv"
        self.remote_catalog_url = remote_catalog_url
        self.cache_path = Path(cache_path) if cache_path else None
        self._load_embedded()

    @staticmethod
    def _clean_row(row: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize CSV/JSON row keys and string values."""
        return {
            (key.strip() if isinstance(key, str) else key): (value.strip() if isinstance(value, str) else value)
            for key, value in row.items()
            if key is not None
        }

    def _append_rows(self, rows: Iterable[Dict[str, Any]]) -> None:
        """Append normalized rows to the catalog."""
        for row in rows:
            self.entries.append(CatalogEntry(**self._clean_row(row)))

    def _load_embedded(self) -> None:
        """Load embedded seed data from package resources or repo-relative CSV."""
        # First try package resource (works for installed packages)
        try:
            data = pkgutil.get_data(__package__.split(".")[0], self.embedded_path)
        except Exception:
            data = None

        if data:
            try:
                text = data.decode("utf-8")
                reader = csv.DictReader(io.StringIO(text))
                self._append_rows(reader)
                return
            except Exception:
                # If package resource present but parsing failed, continue to fallback
                pass

        # Fallback: search upward from this file for a data directory containing the embedded CSV
        try:
            current = Path(__file__).resolve().parent
            # walk up a reasonable number of parents looking for the data file
            for parent in [current] + list(current.parents)[:8]:
                # try the exact embedded path first
                candidate = parent / self.embedded_path
                if candidate.is_file():
                    with candidate.open("r", encoding="utf-8") as fh:
                        reader = csv.DictReader(fh)
                        self._append_rows(reader)
                    return
                # also try common layout where 'data/' is at repo root
                candidate2 = parent / "data" / Path(self.embedded_path).name
                if candidate2.is_file():
                    with candidate2.open("r", encoding="utf-8") as fh:
                        reader = csv.DictReader(fh)
                        self._append_rows(reader)
                    return
        except Exception:
            # Embedded data not found or failed to load; catalog remains empty
            pass

    def load_user_csv(self, csv_path: str) -> None:
        """Load user-provided CSV rows and append them to the catalog."""
        with open(csv_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            self._append_rows(reader)

    def query(
        self,
        capacity_kW: float,
        refrigerant_family: Optional[str] = None,
        effect_type: Optional[str] = None,
    ) -> List[CatalogEntry]:
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

    @staticmethod
    def _entry_matches(
        entry: CatalogEntry,
        refrigerant_family: Optional[str] = None,
        effect_type: Optional[str] = None,
    ) -> bool:
        if refrigerant_family and entry.get("refrigerant_family") != refrigerant_family:
            return False
        if effect_type and entry.get("effect_type") != effect_type:
            return False
        return True

    def provenance_issues(self) -> List[str]:
        """Return human-readable provenance issues for catalog entries.

        The default seed catalog should include enough provenance metadata for
        users to understand that values are estimates derived from public
        manufacturer literature rather than licensed vendor databases.
        """
        issues: List[str] = []
        for index, entry in enumerate(self.entries, start=1):
            model_id = entry.get("model_id", f"row {index}")
            for column in PROVENANCE_COLUMNS:
                if not entry.get(column):
                    issues.append(f"{model_id}: missing {column}")
        return issues

    def has_complete_provenance(self) -> bool:
        """Return ``True`` when every row has required provenance columns."""
        return len(self.provenance_issues()) == 0

    def select_min_cost_set(
        self,
        required_capacity_kW: float,
        allow_staging: bool = True,
        refrigerant_family: Optional[str] = None,
        effect_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a selected set of units and estimated cost.

        This is a naive greedy packer by descending nominal size that tries to
        reach the required capacity.
        """
        # Try to use an integer linear program to minimize installed cost while meeting capacity
        entries = [
            entry
            for entry in self.entries
            if self._entry_matches(entry, refrigerant_family=refrigerant_family, effect_type=effect_type)
        ]
        if not entries:
            return {"selected": [], "total_capacity_kW": 0.0, "estimated_cost_USD": 0.0}

        # Prepare data
        caps = []
        costs = []
        ids = []
        for e in entries:
            try:
                caps.append(float(e.get("nominal_cooling_kW", 0)))
            except Exception:
                caps.append(0.0)
            try:
                costs.append(float(e.get("installed_cost_USD", 0)))
            except Exception:
                costs.append(0.0)
            ids.append(e.get("model_id"))

        # If PuLP is available, solve an integer knapsack-like problem. Otherwise fall back to greedy.
        try:
            import pulp

            prob = pulp.LpProblem("catalog_selection", pulp.LpMinimize)
            var_names = [f"n_{i}" for i in range(len(entries))]
            # upper bounds: at most ceil(required / cap) + 5 as reasonable cap
            ub = [max(0, int((required_capacity_kW // (c if c > 0 else 1)) + 5)) for c in caps]
            vars_dict = {i: pulp.LpVariable(var_names[i], lowBound=0, upBound=ub[i], cat=pulp.LpInteger) for i in range(len(entries))}
            # objective: minimize installed cost
            prob += pulp.lpSum([costs[i] * vars_dict[i] for i in range(len(entries))])
            # capacity constraint
            prob += pulp.lpSum([caps[i] * vars_dict[i] for i in range(len(entries))]) >= float(required_capacity_kW)
            # solve
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
            selection = []
            total_capacity = 0.0
            total_cost = 0.0
            for i in range(len(entries)):
                val = int(pulp.value(vars_dict[i])) if pulp.value(vars_dict[i]) is not None else 0
                if val > 0:
                    selection.append({"model_id": ids[i], "count": val, "nominal_kW": caps[i]})
                    total_capacity += caps[i] * val
                    total_cost += costs[i] * val
            return {"selected": selection, "total_capacity_kW": total_capacity, "estimated_cost_USD": total_cost}
        except Exception:
            # Fall back to greedy packer by descending nominal size
            sorted_entries = sorted(entries, key=lambda e: float(e.get("nominal_cooling_kW", 0)), reverse=True)
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
        """Query a remote JSON or CSV catalog endpoint.

        Parameters
        ----------
        query_params:
            Query-string parameters. The optional keys ``url`` or
            ``remote_catalog_url`` override the catalog's configured default
            endpoint for this call.
        timeout_s:
            Network timeout in seconds.

        Returns
        -------
        list[CatalogEntry]
            Remote catalog entries. Results are not appended to ``entries``
            automatically; callers can decide whether to merge them.

        Notes
        -----
        Supported response formats are a JSON list of row dictionaries, a JSON
        object with an ``entries`` list, or CSV text with catalog-like columns.
        On failure, a previously written cache is returned when available;
        otherwise an empty list is returned.
        """
        endpoint = query_params.get("url") or query_params.get("remote_catalog_url") or self.remote_catalog_url
        if not endpoint:
            return self._load_remote_cache()

        request_params = {
            str(key): value
            for key, value in query_params.items()
            if key not in {"url", "remote_catalog_url"} and value is not None
        }
        url = self._url_with_query(str(endpoint), request_params)
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as response:
                content_type = response.headers.get("Content-Type", "")
                payload = response.read().decode("utf-8-sig")
            entries = self._parse_remote_payload(payload, content_type)
            if entries:
                self._write_remote_cache(entries)
            return entries
        except Exception:
            return self._load_remote_cache()

    @staticmethod
    def _url_with_query(url: str, params: Dict[str, Any]) -> str:
        """Return ``url`` with ``params`` appended to its query string."""
        if not params:
            return url
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        query.extend((key, str(value)) for key, value in params.items())
        return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query)))

    def _parse_remote_payload(self, payload: str, content_type: str = "") -> List[CatalogEntry]:
        """Parse remote JSON or CSV payload text into catalog entries."""
        stripped = payload.lstrip()
        rows: List[Dict[str, Any]]
        if "json" in content_type.lower() or stripped.startswith(("[", "{")):
            decoded = json.loads(payload)
            if isinstance(decoded, dict):
                decoded = decoded.get("entries", [])
            if not isinstance(decoded, list):
                return []
            rows = [row for row in decoded if isinstance(row, dict)]
        else:
            rows = list(csv.DictReader(io.StringIO(payload)))
        return [CatalogEntry(**self._clean_row(row)) for row in rows]

    def _write_remote_cache(self, entries: List[CatalogEntry]) -> None:
        """Write remote results to the optional JSON cache."""
        if self.cache_path is None:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("w", encoding="utf-8") as cache_file:
                json.dump([entry.to_dict() for entry in entries], cache_file, indent=2)
        except Exception:
            pass

    def _load_remote_cache(self) -> List[CatalogEntry]:
        """Load cached remote entries when a cache path exists and is readable."""
        if self.cache_path is None or not self.cache_path.is_file():
            return []
        try:
            with self.cache_path.open("r", encoding="utf-8") as cache_file:
                cached = json.load(cache_file)
            if not isinstance(cached, list):
                return []
            return [CatalogEntry(**self._clean_row(row)) for row in cached if isinstance(row, dict)]
        except Exception:
            return []
