"""``python -m calliopy.roiquest``: open RoiQuest on an adata file or an ExpID."""

from __future__ import annotations

import argparse
import logging


def main(argv=None) -> int:
    """Command line entry point (see ``python -m calliopy.roiquest --help``)."""
    p = argparse.ArgumentParser(
        prog="python -m calliopy.roiquest",
        description="RoiQuest: draw and curate ROIs on two-photon and widefield adata files.")
    p.add_argument("adata_path", nargs="?", help="adata file to open (file dialog if omitted)")
    p.add_argument("--original", metavar="FILE",
                   help="adata file of the original recording of the chronic series")
    p.add_argument("--templates", metavar="DIR", help="folder of widefield ROI templates")
    p.add_argument("--expid", type=int, help="open an ExpID (needs --explog and --adata-dir)")
    p.add_argument("--explog", metavar="FILE", help="offline ExpLog.mat")
    p.add_argument("--adata-dir", metavar="DIR", help="root folder of the adata files")
    p.add_argument("--quiet", action="store_true", help="do not log RoiQuest messages")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="[%(asctime)s] - %(message)s", datefmt="%H:%M:%S")
    exp_log = None
    if args.explog:
        from calliopy.explog import get_exp_log

        exp_log = get_exp_log(explog_file=args.explog)
    if args.expid is not None and (exp_log is None or not args.adata_dir):
        p.error("--expid needs --explog and --adata-dir")

    from . import run

    run(args.adata_path, expid=args.expid, exp_log=exp_log, adata_dir=args.adata_dir,
        original_adata_path=args.original, roi_template_dir=args.templates, block=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
