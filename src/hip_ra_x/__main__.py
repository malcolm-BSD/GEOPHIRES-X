from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from geophires_x import __version__ as geophires_x_version
from hip_ra import HipRaInputParameters
from hip_ra_x import HipRaXClient
from hip_ra_x import HipRaXResult


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HIP-RA-X CLI")
    parser.add_argument("input_file", type=Path, help="HIP-RA-X input file path")
    parser.add_argument("--output-file", type=Path, help="Write the text case report to this path")
    parser.add_argument("--json-output-file", type=Path, help="Write a machine-readable result envelope to this path")
    parser.add_argument("--no-console", action="store_true", help="Do not print the case report to standard output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    input_file = args.input_file.absolute()

    if not input_file.is_file():
        raise FileNotFoundError(f"HIP-RA-X input file does not exist: {input_file}")

    output_file = args.output_file.absolute() if args.output_file is not None else None
    json_output_file = args.json_output_file.absolute() if args.json_output_file is not None else None

    for requested_output in (output_file, json_output_file):
        if requested_output is not None and not requested_output.parent.is_dir():
            raise FileNotFoundError(f"Output directory does not exist: {requested_output.parent}")

    stash_cwd = Path.cwd()
    stash_sys_argv = sys.argv

    try:
        with tempfile.TemporaryDirectory(prefix="hip-ra-x-") as temp_dir:
            augmented_input_file = Path(temp_dir, input_file.name)
            input_file_lines = input_file.read_text(encoding="utf-8").splitlines()
            input_file_lines.append(f"Print Output to Console, {not args.no_console}")
            augmented_input_file.write_text("\n".join(input_file_lines) + "\n", encoding="utf-8")

            input_params = HipRaInputParameters(augmented_input_file, output_file_path=output_file)
            result: HipRaXResult = HipRaXClient().get_hip_ra_x_result(input_params)

            if json_output_file is not None:
                result_envelope = {
                    "schema_version": 1,
                    "model": {"id": "hip-ra-x", "version": geophires_x_version},
                    "result": result.result,
                    "case_report": result.caseReportText,
                }
                json_output_file.write_text(
                    json.dumps(result_envelope, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
    finally:
        # The legacy model changes process-global state internally. The CLI restores it for callers that invoke main().
        sys.argv = stash_sys_argv
        os.chdir(stash_cwd)

    return 0


if __name__ == "__main__":
    sys.exit(main())
