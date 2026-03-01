from __future__ import annotations

import argparse
import sys
from pathlib import Path

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download YouTube videos + captions (requires rights/permission).")
    p.add_argument("url", help="YouTube video URL")
    p.add_argument("--out", default="./downloads", help="Output directory (default: ./downloads)")
    p.add_argument(
        "--lang",
        default="en",
        help='Subtitle language(s), comma-separated (default: "en"). Example: "en,es"',
    )
    p.add_argument(
        "--auto",
        action="store_true",
        help="Download auto-generated captions too (if available).",
    )
    p.add_argument(
        "--no-video",
        action="store_true",
        help="Only download captions (skip video).",
    )
    p.add_argument(
        "--srt",
        action="store_true",
        help="Convert subtitles to .srt when possible (requires ffmpeg).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # yt-dlp output template:
    # downloads/<title> [<id>].<ext>
    outtmpl = str(out_dir / "%(title)s [%(id)s].%(ext)s")

    langs = [x.strip() for x in args.lang.split(",") if x.strip()]

    ydl_opts = {
        "outtmpl": outtmpl,
        "noplaylist": True,  # set to False if you want playlist support
        "writesubtitles": True,
        "writeautomaticsub": bool(args.auto),
        "subtitleslangs": langs,
        "subtitlesformat": "best",
        "quiet": False,
        "no_warnings": False,
    }

    if args.no_video:
        # Skip video download; only fetch subs/metadata
        ydl_opts["skip_download"] = True
    else:
        # Prefer MP4 if possible, otherwise best available
        ydl_opts["format"] = "bv*+ba/best"

        # If you strictly want MP4 when available:
        # ydl_opts["format"] = "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best"

    if args.srt:
        # Convert subtitle files to .srt (requires ffmpeg)
        ydl_opts["postprocessors"] = [
            {"key": "FFmpegSubtitlesConvertor", "format": "srt"},
        ]

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([args.url])
        return 0
    except DownloadError as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
