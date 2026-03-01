"""Script to automate naming TV show files for Plex, using TVDB as the source of truth for show/season/episode metadata."""

from __future__ import annotations  # Until Python 3.14

import os
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import typer
from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console()

_TVDB_BASE = "https://api4.thetvdb.com/v4"

_VIDEO_EXTS = {".mkv", ".mp4", ".m4v", ".avi", ".mov"}

_PATTERNS_TV = [
    re.compile(r"(?i)\bS(?P<season>\d{1,2})E(?P<ep>\d{1,2})\b"),  # S01E02
    re.compile(r"(?i)\b(?P<season>\d{1,2})x(?P<ep>\d{2})\b"),  # 1x02
    re.compile(r"(?i)\bSeason[ ._-]?(?P<season>\d{1,2}).*?\bEp(?:isode)?[ ._-]?(?P<ep>\d{1,3})\b"),
]

_PATTERN_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _sanitize(s: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1F]', "", s).strip()


def _normalize_title(title: str) -> str:
    return unicodedata.normalize("NFKC", title).strip()


def _ensure_unique_path(p: Path) -> Path:
    if not p.exists():
        return p
    stem, suf = p.stem, p.suffix
    for i in range(2, 5000):
        candidate = p.with_name(f"{stem} ({i}){suf}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Too many collisions for {p}")


def _list_video_files(folder: Path) -> list[Path]:
    files: list[Path] = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTS:
            files.append(p)
    return files


def _infer_season_episode_from_filename(name: str) -> tuple[int, int] | None:
    for pat in _PATTERNS_TV:
        m = pat.search(name)
        if m:
            season = int(m.group("season"))
            ep = int(m.group("ep"))
            return season, ep
    return None


def _guess_year_from_text(text: str) -> int | None:
    m = _PATTERN_YEAR.search(text)
    return int(m.group(1)) if m else None


def _extract_english_title(obj: dict[str, Any]) -> str | None:
    translations = obj.get("translations") or {}
    eng = translations.get("eng")
    if eng and isinstance(eng, str) and eng.strip():
        return eng.strip()
    return None


@dataclass
class TvdbClient:
    api_key: str
    token: str | None = None

    # Keep a single http client to reuse connections
    _client: httpx.Client | None = None

    def __enter__(self) -> "TvdbClient":
        self._client = httpx.Client(timeout=30)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._client:
            self._client.close()
        self._client = None

    def _headers(self) -> dict[str, str]:
        h = {
            "Accept": "application/json",
            "Accept-Language": "eng",  # English-first at the transport level
        }
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _get(self, url: str, *, params: dict[str, str] | None = None) -> httpx.Response:
        if not self._client:
            raise RuntimeError("TvdbClient must be used as a context manager.")
        return self._client.get(url, params=params, headers=self._headers())

    def _post(self, url: str, *, json: dict[str, Any]) -> httpx.Response:
        if not self._client:
            raise RuntimeError("TvdbClient must be used as a context manager.")
        return self._client.post(url, json=json, headers=self._headers())

    def login(self) -> None:
        r = self._post(f"{_TVDB_BASE}/login", json={"apikey": self.api_key})
        r.raise_for_status()
        self.token = r.json()["data"]["token"]

    def search_series(self, query: str, year: int | None = None) -> list[dict[str, Any]]:
        params: dict[str, str] = {"q": query, "type": "series"}
        if year:
            params["year"] = str(year)
        r = self._get(f"{_TVDB_BASE}/search", params=params)
        r.raise_for_status()
        return r.json().get("data") or []

    def series_extended(self, series_id: int) -> dict[str, Any]:
        r = self._get(f"{_TVDB_BASE}/series/{series_id}/extended")
        r.raise_for_status()
        return r.json()["data"]

    def series_episodes_by_season_type(
        self,
        series_id: int,
        season_type: str = "official",
        page: int = 0,
    ) -> dict[str, Any]:
        # Returns JSON with data + links.pagination (varies), so we return the full payload.
        params = {"page": str(page)}
        r = self._get(f"{_TVDB_BASE}/series/{series_id}/episodes/{season_type}", params=params)
        r.raise_for_status()
        return r.json()

    def build_episode_index(
        self,
        series_id: int,
        season_type: str = "official",
    ) -> dict[tuple[int, int], int]:
        """Build a mapping: {(season, episode): episode_id} for a given series and season type, by paginating through the TVDB API.

        Args:
            series_id (int): The ID of the series.
            season_type (str, optional): The type of season to retrieve. Defaults to "official".

        Returns:
            dict[tuple[int, int], int]: A mapping of (season, episode) to episode_id.

        """
        index: dict[tuple[int, int], int] = {}

        page = 0
        while True:
            payload = self.series_episodes_by_season_type(series_id, season_type=season_type, page=page)
            items = payload.get("data") or []
            for ep in items.get("episodes", []):
                # TVDB episode list objects commonly include:
                # id, airedSeason, airedEpisode, etc.
                try:
                    aired_season = ep.get("seasonNumber")
                    aired_episode = ep.get("number")
                    ep_id = ep.get("id")
                    if isinstance(aired_season, int) and isinstance(aired_episode, int) and isinstance(ep_id, int):
                        index[(aired_season, aired_episode)] = ep_id
                except Exception:
                    continue

            links = payload.get("links") or {}
            next_page = links.get("next")
            # TVDB uses 0-based pages; if next is null/None, we're done.
            if next_page is None:
                break
            page = int(next_page)

        return index

    def episode_by_id(self, episode_id: int) -> dict[str, Any]:
        r = self._get(f"{_TVDB_BASE}/episodes/{episode_id}")
        r.raise_for_status()
        return r.json().get("data") or {}

    def episode_translation(self, episode_id: int, language: str = "eng") -> dict[str, Any] | None:
        r = self._get(f"{_TVDB_BASE}/episodes/{episode_id}/translations/{language}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json().get("data") or None

    def get_episode_title_english_first(self, episode_id: int) -> str | None:
        # 1) translation endpoint (explicit English)
        trans = self.episode_translation(episode_id, "eng")
        if trans:
            name = trans.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()

        # 2) fallback to episode details
        ep = self.episode_by_id(episode_id)
        for key in ("name", "episodeName"):
            val = ep.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        return None


def _choose_show(hits: list[dict[str, Any]]) -> dict[str, Any]:
    table = Table(title="TVDB Search Results (English-first)")
    table.add_column("#", justify="right")
    table.add_column("English Title")
    table.add_column("Default Title")
    table.add_column("Year", justify="right")
    table.add_column("TVDB ID", justify="right")

    for i, h in enumerate(hits, start=1):
        eng = _extract_english_title(h) or ""
        default_name = str(h.get("name") or "")
        table.add_row(
            str(i),
            _normalize_title(eng) if eng else "-",
            _normalize_title(default_name) if default_name else "-",
            str(h.get("year") or ""),
            str(h.get("tvdb_id") or ""),
        )

    console.print(table)
    idx = typer.prompt("Choose a show number", type=int)
    if idx < 1 or idx > len(hits):
        raise typer.BadParameter("Selection out of range.")
    return hits[idx - 1]


def _plex_episode_filename(show_folder: str, season: int, episode: int, ep_title: str, ext: str) -> str:
    return f"{show_folder} - s{season:02d}e{episode:02d} - {_sanitize(_normalize_title(ep_title))}{ext}"


def _plex_episode_dest(tv_root: Path, show_folder: str, season: int, filename: str) -> Path:
    return tv_root / show_folder / f"Season {season:02d}" / filename


def _prompt_mode() -> str:
    print("\nModes:")
    print("  1) auto    = infer S/E and rename without asking per-file")
    print("  2) confirm = infer S/E and proposed name, ask Y/N per-file")
    print("  3) manual  = ask for season/episode for each file")
    choice = typer.prompt("Choose mode (1/2/3)", type=int)
    if choice == 1:
        return "auto"
    if choice == 2:
        return "confirm"
    if choice == 3:
        return "manual"
    raise typer.BadParameter("Invalid mode.")


@app.command()
def run(
    folder: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, help="Folder containing episode files"),
    tv_root: Path = typer.Option(Path("./TV"), help="Destination TV root folder"),
    season_type: str = typer.Option(
        "official",
        help="TVDB season type (commonly: official, dvd, absolute, alternate, regional)",
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help="Preview changes without moving files",
    ),
):
    # Load the API key from environment or .env file
    load_dotenv()
    api_key = os.getenv("TVDB_API_KEY")
    if not api_key:
        raise typer.BadParameter("Set TVDB_API_KEY (env var or .env).")
    # Login to the TVDB API
    with TvdbClient(api_key=api_key) as client:
        client.login()
        # Ask the user for the show title
        show_query = typer.prompt("TV show name (raw text)")
        # Try to guess the year from what they provided, to improve search relevance (but it's optional)
        year_hint = _guess_year_from_text(show_query)
        # Search TVDB for the show
        hits = client.search_series(show_query, year=year_hint)
        # If we couldn't find anything, exit out with error
        if not hits:
            print(f"[red]No TVDB results for:[/] {show_query}")
            raise typer.Exit(code=1)
        # Prompt the user for the show they want from the search results (if there's more than one), and get the series ID
        selected = _choose_show(hits)
        series_id = int(selected["tvdb_id"])
        series = client.series_extended(series_id)
        # English-first show name (translations mapping), then fallbacks
        show_name_raw = _extract_english_title(series) or _extract_english_title(selected) or series.get("name") or selected.get("name") or show_query
        show_name = _sanitize(_normalize_title(str(show_name_raw)))
        # Get metadata about the show to create the show folder name, e.g. "Show Name (2020) {tvdb-12345}"
        first_aired = str(series.get("firstAired") or "")
        show_year = first_aired[:4] if len(first_aired) >= 4 else str(selected.get("year") or "").strip()
        show_folder = f"{show_name} ({show_year}) {{tvdb-{series_id}}}" if show_year.isdigit() else show_name
        # Then prompt the user for the mode they want to use for renaming (auto, confirm, manual)
        mode = _prompt_mode()
        files = _list_video_files(folder)
        if not files:
            print(f"[yellow]No video files found in[/] {folder}")
            raise typer.Exit(code=0)
        print(f"\nSelected: [bold]{show_folder}[/bold] (TVDB {series_id})")
        print(f"Found {len(files)} files in {folder}")
        print(f"Destination root: {tv_root}")
        print(f"Season type: {season_type}")
        print(f"Dry run: {dry_run}\n")
        # Build once: (airedSeason, airedEpisode) -> episode_id index
        episode_index = client.build_episode_index(series_id, season_type=season_type)
        # For each file, rename according to the selected mode
        for f in files:
            season_ep: tuple[int, int] | None = None
            # If auto/confirm mode, try to infer season/episode from the filename; if we can't, skip (auto) or ask (confirm)
            if mode in ("auto", "confirm"):
                season_ep = _infer_season_episode_from_filename(f.name)
                if not season_ep:
                    print(f"[yellow]Could not infer S/E from filename:[/] {f.name}")
                    if mode == "auto":
                        continue
            # If the mode is manual, or if we're in confirm mode and couldn't infer S/E, prompt the user for season/episode numbers
            if mode == "manual" or (mode == "confirm" and not season_ep):
                season = typer.prompt(f"{f.name} - season #", type=int)
                episode = typer.prompt(f"{f.name} - episode #", type=int)
                season_ep = (season, episode)
            # Get the episode ID from our index
            season, episode = season_ep
            ep_id = episode_index.get((season, episode))
            if not ep_id:
                print(f"[yellow]No episode id found for[/] {show_folder} s{season:02d}e{episode:02d} (season_type={season_type})")
                # In confirm/manual modes, you might want to keep going; in auto mode, we just skip.
                continue
            # Get the episode title (English-first) from the TVDB API, and construct the new filename and destination path
            ep_title = client.get_episode_title_english_first(ep_id) or f"Episode {episode:02d}"
            new_name = _plex_episode_filename(show_folder, season, episode, ep_title, f.suffix.lower())
            dest = _plex_episode_dest(tv_root, show_folder, season, new_name)
            dest = _ensure_unique_path(dest)
            # Confirm if in confirm mode
            if mode == "confirm":
                print(f"[cyan]Proposed[/] {f.name} -> {dest}")
                ok = typer.confirm("Rename/move this file?", default=True)
                if not ok:
                    continue
            # Show dry run, or actually move the file if not a dry run
            if dry_run:
                print(f"[cyan]DRY RUN[/] {f} -> {dest}")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(dest))
                print(f"[green]MOVED[/] {f} -> {dest}")
        # Done
        print("\n[bold green]Done.[/bold green]")


if __name__ == "__main__":
    app()
