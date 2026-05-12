# Absorption Chiller Catalog Seed Data

`absorption_chiller_catalog_default.csv` is a GEOPHIRES seed catalog for
screening studies. It is not a procurement catalog and should not be treated as
vendor-certified equipment data.

## Provenance Policy

Each row carries:

- `source`: short description of the public literature used to set or bound the
  seed values.
- `source_url`: public manufacturer or manual page reviewed for the row.
- `last_verified`: date the public source was checked.
- `license_note`: usage/safety note for downstream users.

The numeric values are public-literature-derived engineering estimates. Some
rows use vendor category pages to anchor technology family and operating range,
with capacity/cost/footprint values retained as GEOPHIRES seed estimates. Users
who need procurement-grade results should replace this CSV with vendor quotes or
submittals for their project conditions.

## Remote Catalogs

`geophires_x.absorption.catalog.Catalog.query_remote_catalog` can read JSON or
CSV endpoints and optionally cache successful responses. Remote rows should use
the same provenance columns when possible. Remote scraping must respect site
terms; prefer manufacturer-provided CSV or JSON endpoints.
