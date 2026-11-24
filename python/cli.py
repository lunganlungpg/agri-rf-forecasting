"""
Command-line entry point installed as ``agri-rf``.

Examples::

    agri-rf forecast                      # run the CV pipeline on the sample data
    agri-rf forecast --data my_farm.csv   # ... or on your own CSV
    agri-rf plots --out figures/          # static PNG figures
    agri-rf interactive --out figures/    # Plotly HTML dashboards
"""

import argparse

from . import __version__
from .agri_rf_forecast import run_pipeline
from .generate_interactive_plots import build_interactive_dashboard
from .generate_plots import create_plots


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="agri-rf",
        description="Random Forest forecasting for agricultural yield time series.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # Every sub-command accepts an optional CSV; plotting ones also an output folder.
    p_fc = sub.add_parser("forecast", help="Time-series cross-validation + feature importance")
    p_fc.add_argument("--data", help="CSV file or URL (default: bundled sample data)")

    for name, helptext in (
        ("plots", "Write 300 DPI PNG figures"),
        ("interactive", "Write interactive Plotly HTML dashboards"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--data", help="CSV file or URL (default: bundled sample data)")
        p.add_argument("--out", help="Output folder (default: ./plots)")

    args = parser.parse_args(argv)
    if args.command == "forecast":
        run_pipeline(args.data)
    elif args.command == "plots":
        create_plots(args.data, args.out)
    else:
        build_interactive_dashboard(args.data, args.out)


if __name__ == "__main__":
    main()
