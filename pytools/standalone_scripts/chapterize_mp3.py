from __future__ import annotations

import bisect
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Literal

import typer
from loguru import logger

script = typer.Typer(add_completion=False)

_REGEX_PATTERN_SILENCE_START = re.compile(r"silence_start:\s*(\d+(?:\.\d+)?)")
_REGEX_PATTERN_SILENCE_END = re.compile(r"silence_end:\s*(\d+(?:\.\d+)?)\s*\|\s*silence_duration:\s*(\d+(?:\.\d+)?)")
_REGEX_TOKEN = re.compile(r"[A-Za-z][A-Za-z']+")

_STOP_WORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


@dataclass(slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass(slots=True)
class SilenceInterval:
    start: float
    end: float
    duration: float


@dataclass(slots=True)
class BoundaryCandidate:
    time_s: float
    score: float


def time_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    start = monotonic()
    result = fn(*args, **kwargs)
    elapsed = monotonic() - start
    return result, elapsed


@script.command()
def chapterize_mp3(
    input_path: Path = typer.Argument(..., exists=True, help="MP3 file or directory containing MP3 files."),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively scan subdirectories for MP3s."),
    mode: Literal["hybrid", "silence", "transcript"] = typer.Option(
        "hybrid",
        help="Boundary detection mode: hybrid(content + silence), silence-only, or transcript-only.",
    ),
    target_chapter_minutes: float = typer.Option(10.0, min=1.0, help="Approximate target chapter length."),
    min_chapter_minutes: float = typer.Option(3.0, min=0.5, help="Minimum allowed chapter length."),
    min_silence_len_s: float = typer.Option(1.5, min=0.1, help="Minimum silence duration for ffmpeg silencedetect."),
    silence_threshold_db: float = typer.Option(-38.0, max=0.0, help="Silence threshold in dB (negative values)."),
    silence_snap_window_s: float = typer.Option(20.0, min=0.0, help="Max distance to snap transcript boundaries to silence."),
    transcript_window_segments: int = typer.Option(4, min=1, help="Segments on each side for topic-shift scoring."),
    transcript_model: str = typer.Option("small.en", help="faster-whisper model name/path for transcript modes."),
    transcript_device: str = typer.Option("auto", help="faster-whisper device, e.g. auto/cpu/cuda."),
    transcript_compute_type: str = typer.Option("int8", help="faster-whisper compute type for speed/quality tradeoff."),
    transcript_cache: bool = typer.Option(True, "--transcript-cache/--no-transcript-cache", help="Reuse cached transcript JSON when present."),
    overwrite_transcript_cache: bool = typer.Option(
        False,
        "--overwrite-transcript-cache",
        help="Regenerate transcript JSON even if cache exists.",
    ),
    title_keywords: int = typer.Option(3, min=1, help="Number of keywords to include in generated titles."),
    threads: int = typer.Option(8, min=1, help="ffmpeg / whisper CPU threads."),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG/INFO/WARNING/ERROR)."),
) -> None:
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    _require_binary("ffmpeg")
    _require_binary("ffprobe")

    mp3_files = _collect_mp3_files(input_path, recursive)
    if not mp3_files:
        raise typer.BadParameter(f"No .mp3 files found at: {input_path}")

    logger.info(f"Found {len(mp3_files)} MP3 file(s).")
    for idx, mp3_file in enumerate(mp3_files, start=1):
        logger.info(f"[{idx}/{len(mp3_files)}] Processing {mp3_file}...")
        _chapterize_single_mp3(
            mp3_filepath=mp3_file,
            mode=mode,
            target_chapter_minutes=target_chapter_minutes,
            min_chapter_minutes=min_chapter_minutes,
            min_silence_len_s=min_silence_len_s,
            silence_threshold_db=silence_threshold_db,
            silence_snap_window_s=silence_snap_window_s,
            transcript_window_segments=transcript_window_segments,
            transcript_model=transcript_model,
            transcript_device=transcript_device,
            transcript_compute_type=transcript_compute_type,
            transcript_cache=transcript_cache,
            overwrite_transcript_cache=overwrite_transcript_cache,
            title_keywords=title_keywords,
            threads=threads,
        )


def _chapterize_single_mp3(
    mp3_filepath: Path,
    mode: Literal["hybrid", "silence", "transcript"],
    target_chapter_minutes: float,
    min_chapter_minutes: float,
    min_silence_len_s: float,
    silence_threshold_db: float,
    silence_snap_window_s: float,
    transcript_window_segments: int,
    transcript_model: str,
    transcript_device: str,
    transcript_compute_type: str,
    transcript_cache: bool,
    overwrite_transcript_cache: bool,
    title_keywords: int,
    threads: int,
) -> None:
    if not mp3_filepath.is_file():
        raise ValueError(f"File not found: {mp3_filepath}")

    outfile = mp3_filepath.with_suffix(".m4b")
    metadata_path = mp3_filepath.with_suffix(".ffmetadata")
    transcript_cache_path = mp3_filepath.with_suffix(".transcript.json")

    duration_s, _ = time_call(_ffprobe_duration_seconds, mp3_filepath)
    source_bitrate = _ffprobe_audio_bitrate(mp3_filepath)
    output_bitrate = min(source_bitrate, 128000)

    logger.info(f"Audio duration: {duration_s / 60.0:.2f} minutes")
    logger.info(f"Source bitrate: {source_bitrate / 1000:.0f} kbps -> output bitrate: {output_bitrate / 1000:.0f} kbps")

    silence_intervals: list[SilenceInterval] = []
    if mode in {"hybrid", "silence"}:
        silence_intervals, silence_elapsed = time_call(
            _ffmpeg_detect_silence_intervals,
            mp3_filepath,
            min_silence_len_s=min_silence_len_s,
            silence_threshold_db=silence_threshold_db,
            threads=threads,
        )
        logger.info(f"Detected {len(silence_intervals)} silence interval(s) in {silence_elapsed:.2f}s")

    transcript_segments: list[TranscriptSegment] = []
    if mode in {"hybrid", "transcript"}:
        transcript_segments, transcript_elapsed = time_call(
            _load_or_transcribe_segments,
            mp3_filepath,
            transcript_cache_path,
            transcript_cache=transcript_cache,
            overwrite_cache=overwrite_transcript_cache,
            model_name=transcript_model,
            device=transcript_device,
            compute_type=transcript_compute_type,
            threads=threads,
        )
        logger.info(f"Transcript segments: {len(transcript_segments)} in {transcript_elapsed:.2f}s")

    min_chapter_len_s = min_chapter_minutes * 60.0
    target_chapter_len_s = target_chapter_minutes * 60.0

    boundaries = _build_boundaries(
        duration_s=duration_s,
        mode=mode,
        transcript_segments=transcript_segments,
        silence_intervals=silence_intervals,
        min_chapter_len_s=min_chapter_len_s,
        target_chapter_len_s=target_chapter_len_s,
        transcript_window_segments=transcript_window_segments,
        silence_snap_window_s=silence_snap_window_s,
    )

    chapters = _boundaries_to_chapters(boundaries)
    chapters = _apply_titles_from_transcript(chapters, transcript_segments, title_keywords=title_keywords)
    logger.info(f"Final chapters: {len(chapters)}")

    _write_ffmetadata(chapters, metadata_path)
    _encode_m4b_with_chapters(
        input_audio=mp3_filepath,
        outfile=outfile,
        metadata_file=metadata_path,
        bitrate=f"{output_bitrate // 1000}k",
        threads=threads,
    )
    metadata_path.unlink(missing_ok=True)
    logger.info(f"Wrote: {outfile}")


def _collect_mp3_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".mp3":
            raise typer.BadParameter(f"Expected an .mp3 file, got: {input_path}")
        return [input_path]

    pattern = "**/*.mp3" if recursive else "*.mp3"
    return sorted(path for path in input_path.glob(pattern) if path.is_file())


def _load_or_transcribe_segments(
    mp3_filepath: Path,
    cache_path: Path,
    transcript_cache: bool,
    overwrite_cache: bool,
    model_name: str,
    device: str,
    compute_type: str,
    threads: int,
) -> list[TranscriptSegment]:
    if transcript_cache and cache_path.exists() and not overwrite_cache:
        logger.info(f"Using cached transcript: {cache_path}")
        return _load_segments_from_json(cache_path)

    segments = _transcribe_with_faster_whisper(
        mp3_filepath=mp3_filepath,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        threads=threads,
    )
    if transcript_cache:
        _write_segments_to_json(cache_path, segments)
    return segments


def _transcribe_with_faster_whisper(
    mp3_filepath: Path,
    model_name: str,
    device: str,
    compute_type: str,
    threads: int,
) -> list[TranscriptSegment]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Transcript mode requires faster-whisper. Install with: pip install faster-whisper"
        ) from exc

    logger.info(f"Transcribing with faster-whisper model='{model_name}', device='{device}', compute_type='{compute_type}'")
    model = WhisperModel(model_name, device=device, compute_type=compute_type, cpu_threads=threads)
    raw_segments, _info = model.transcribe(
        str(mp3_filepath),
        beam_size=1,
        best_of=1,
        temperature=0.0,
        vad_filter=True,
        word_timestamps=False,
        condition_on_previous_text=False,
    )

    segments: list[TranscriptSegment] = []
    for segment in raw_segments:
        text = segment.text.strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                start=float(segment.start),
                end=float(segment.end),
                text=text,
            )
        )
    return segments


def _load_segments_from_json(path: Path) -> list[TranscriptSegment]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("segments", [])
    segments: list[TranscriptSegment] = []
    for item in raw:
        try:
            segments.append(
                TranscriptSegment(
                    start=float(item["start"]),
                    end=float(item["end"]),
                    text=str(item["text"]).strip(),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return segments


def _write_segments_to_json(path: Path, segments: list[TranscriptSegment]) -> None:
    data = {
        "segments": [asdict(segment) for segment in segments],
    }
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


def _build_boundaries(
    duration_s: float,
    mode: Literal["hybrid", "silence", "transcript"],
    transcript_segments: list[TranscriptSegment],
    silence_intervals: list[SilenceInterval],
    min_chapter_len_s: float,
    target_chapter_len_s: float,
    transcript_window_segments: int,
    silence_snap_window_s: float,
) -> list[float]:
    if duration_s <= 0:
        raise ValueError("Audio duration must be > 0")

    if mode == "silence":
        silence_candidates = _silence_candidates(silence_intervals)
        return _select_boundaries_from_candidates(silence_candidates, duration_s, min_chapter_len_s, target_chapter_len_s)

    if not transcript_segments:
        logger.warning("No transcript segments were produced; falling back to silence-only boundaries.")
        silence_candidates = _silence_candidates(silence_intervals)
        return _select_boundaries_from_candidates(silence_candidates, duration_s, min_chapter_len_s, target_chapter_len_s)

    transcript_candidates = _transcript_candidates(
        transcript_segments,
        window_segments=transcript_window_segments,
    )

    if mode == "transcript":
        return _select_boundaries_from_candidates(transcript_candidates, duration_s, min_chapter_len_s, target_chapter_len_s)

    # Hybrid mode: add silence influence to transcript candidates and snap resulting cuts to silence.
    hybrid_candidates = _boost_candidates_near_silence(transcript_candidates, silence_intervals, max_distance_s=silence_snap_window_s)
    boundaries = _select_boundaries_from_candidates(hybrid_candidates, duration_s, min_chapter_len_s, target_chapter_len_s)
    return _snap_boundaries_to_silence(boundaries, silence_intervals, snap_window_s=silence_snap_window_s)


def _silence_candidates(silences: list[SilenceInterval]) -> list[BoundaryCandidate]:
    candidates: list[BoundaryCandidate] = []
    for silence in silences:
        midpoint = (silence.start + silence.end) / 2.0
        candidates.append(BoundaryCandidate(time_s=midpoint, score=max(0.01, silence.duration)))
    return candidates


def _transcript_candidates(segments: list[TranscriptSegment], window_segments: int) -> list[BoundaryCandidate]:
    if len(segments) < 2:
        return []

    token_cache = [_tokenize(segment.text) for segment in segments]
    candidates: list[BoundaryCandidate] = []

    for i in range(len(segments) - 1):
        left_slice_start = max(0, i - window_segments + 1)
        right_slice_end = min(len(segments), i + 1 + window_segments)

        left_tokens = set(_flatten_tokens(token_cache[left_slice_start : i + 1]))
        right_tokens = set(_flatten_tokens(token_cache[i + 1 : right_slice_end]))
        similarity = _jaccard_similarity(left_tokens, right_tokens)

        gap_s = max(0.0, segments[i + 1].start - segments[i].end)
        gap_score = min(1.0, gap_s / 2.5)
        topic_shift_score = 1.0 - similarity

        # Topic shift is primary signal, pause gap acts as a secondary signal.
        score = topic_shift_score * 0.8 + gap_score * 0.2
        midpoint = (segments[i].end + segments[i + 1].start) / 2.0
        candidates.append(BoundaryCandidate(time_s=midpoint, score=score))

    return candidates


def _boost_candidates_near_silence(
    candidates: list[BoundaryCandidate],
    silences: list[SilenceInterval],
    max_distance_s: float,
) -> list[BoundaryCandidate]:
    if not candidates or not silences or max_distance_s <= 0:
        return candidates

    mids_and_durations = [((silence.start + silence.end) / 2.0, silence.duration) for silence in silences]
    boosted: list[BoundaryCandidate] = []

    for candidate in candidates:
        nearest_boost = 0.0
        for silence_mid, silence_duration in mids_and_durations:
            distance = abs(candidate.time_s - silence_mid)
            if distance > max_distance_s:
                continue
            closeness = 1.0 - (distance / max_distance_s)
            boost = closeness * min(0.5, silence_duration / 4.0)
            nearest_boost = max(nearest_boost, boost)
        boosted.append(BoundaryCandidate(time_s=candidate.time_s, score=candidate.score + nearest_boost))

    return boosted


def _select_boundaries_from_candidates(
    candidates: list[BoundaryCandidate],
    duration_s: float,
    min_chapter_len_s: float,
    target_chapter_len_s: float,
) -> list[float]:
    if duration_s <= min_chapter_len_s:
        return [0.0, duration_s]

    estimated_chapters = max(1, round(duration_s / target_chapter_len_s))
    target_cuts = max(0, estimated_chapters - 1)

    boundaries = [0.0, duration_s]
    if target_cuts == 0 or not candidates:
        return boundaries

    # Try high-confidence candidates first.
    candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
    for candidate in candidates:
        if len(boundaries) - 2 >= target_cuts:
            break
        inserted = _insert_boundary_if_valid(boundaries, candidate.time_s, min_chapter_len_s)
        if not inserted:
            continue

    # If we still need cuts, place evenly in the largest current chapters.
    while len(boundaries) - 2 < target_cuts:
        longest_index = _index_of_longest_gap(boundaries)
        left = boundaries[longest_index]
        right = boundaries[longest_index + 1]
        midpoint = (left + right) / 2.0
        if not _insert_boundary_if_valid(boundaries, midpoint, min_chapter_len_s):
            break

    return sorted(boundaries)


def _insert_boundary_if_valid(boundaries: list[float], candidate_s: float, min_gap_s: float) -> bool:
    if candidate_s <= 0:
        return False

    pos = bisect.bisect_left(boundaries, candidate_s)
    if pos == 0 or pos >= len(boundaries):
        return False

    left = boundaries[pos - 1]
    right = boundaries[pos]
    if candidate_s - left < min_gap_s:
        return False
    if right - candidate_s < min_gap_s:
        return False

    boundaries.insert(pos, candidate_s)
    return True


def _index_of_longest_gap(boundaries: list[float]) -> int:
    longest = -1.0
    longest_index = 0
    for i in range(len(boundaries) - 1):
        gap = boundaries[i + 1] - boundaries[i]
        if gap > longest:
            longest = gap
            longest_index = i
    return longest_index


def _snap_boundaries_to_silence(
    boundaries: list[float],
    silences: list[SilenceInterval],
    snap_window_s: float,
) -> list[float]:
    if len(boundaries) <= 2 or not silences or snap_window_s <= 0:
        return boundaries

    silence_midpoints = sorted((silence.start + silence.end) / 2.0 for silence in silences)
    result = [boundaries[0]]

    for boundary in boundaries[1:-1]:
        nearest = min(silence_midpoints, key=lambda midpoint: abs(midpoint - boundary), default=boundary)
        if abs(nearest - boundary) <= snap_window_s:
            result.append(nearest)
        else:
            result.append(boundary)

    result.append(boundaries[-1])
    result = sorted(result)

    # Remove accidental duplicate cut points introduced by snapping.
    deduped = [result[0]]
    for boundary in result[1:]:
        if abs(boundary - deduped[-1]) >= 0.25:
            deduped.append(boundary)

    if deduped[-1] != result[-1]:
        deduped.append(result[-1])

    return deduped


def _boundaries_to_chapters(boundaries: list[float]) -> list[tuple[float, float, str]]:
    chapters: list[tuple[float, float, str]] = []
    for idx, (start, end) in enumerate(zip(boundaries, boundaries[1:]), start=1):
        if end - start <= 0.5:
            continue
        chapters.append((start, end, f"Chapter {idx}"))

    if not chapters and boundaries:
        chapters = [(0.0, boundaries[-1], "Chapter 1")]

    return chapters


def _apply_titles_from_transcript(
    chapters: list[tuple[float, float, str]],
    segments: list[TranscriptSegment],
    title_keywords: int,
) -> list[tuple[float, float, str]]:
    if not chapters:
        return chapters

    if not segments:
        return chapters

    titled_chapters: list[tuple[float, float, str]] = []
    for idx, (chapter_start, chapter_end, _default_title) in enumerate(chapters, start=1):
        chapter_text = " ".join(
            segment.text for segment in segments if segment.start < chapter_end and segment.end > chapter_start
        )
        keywords = _top_keywords(chapter_text, limit=title_keywords)
        if keywords:
            title = f"Chapter {idx}: {' / '.join(word.capitalize() for word in keywords)}"
        else:
            title = f"Chapter {idx}"
        titled_chapters.append((chapter_start, chapter_end, title))

    return titled_chapters


def _top_keywords(text: str, limit: int) -> list[str]:
    tokens = _tokenize(text)
    if not tokens:
        return []
    counts = Counter(tokens)
    return [word for word, _count in counts.most_common(limit)]


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_word in _REGEX_TOKEN.findall(text.lower()):
        word = raw_word.strip("'")
        if len(word) < 4:
            continue
        if word in _STOP_WORDS:
            continue
        tokens.append(word)
    return tokens


def _flatten_tokens(token_lists: list[list[str]]) -> list[str]:
    flattened: list[str] = []
    for token_list in token_lists:
        flattened.extend(token_list)
    return flattened


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _ffmpeg_detect_silence_intervals(
    audio_filepath: Path,
    min_silence_len_s: float,
    silence_threshold_db: float,
    threads: int,
) -> list[SilenceInterval]:
    if not audio_filepath.is_file():
        raise ValueError(f"File not found: {audio_filepath}")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-threads",
        str(threads),
        "-i",
        str(audio_filepath),
        "-af",
        f"silencedetect=noise={silence_threshold_db}dB:d={min_silence_len_s}",
        "-f",
        "null",
        "-",
    ]

    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg silencedetect failed:\n{proc.stderr}")

    intervals: list[SilenceInterval] = []
    current_start: float | None = None
    for line in proc.stderr.splitlines():
        start_match = _REGEX_PATTERN_SILENCE_START.search(line)
        if start_match:
            current_start = float(start_match.group(1))
            continue

        end_match = _REGEX_PATTERN_SILENCE_END.search(line)
        if end_match and current_start is not None:
            end = float(end_match.group(1))
            duration = float(end_match.group(2))
            intervals.append(SilenceInterval(start=current_start, end=end, duration=duration))
            current_start = None

    return intervals


def _ffprobe_audio_bitrate(mp3_filepath: Path, default: int = 128000) -> int:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=bit_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(mp3_filepath),
    ]
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if cp.returncode != 0 or not cp.stdout.strip():
        return default
    try:
        return int(cp.stdout.strip())
    except ValueError:
        return default


def _ffprobe_duration_seconds(mp3_filepath: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(mp3_filepath),
    ]
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{cp.stderr}")
    return float(cp.stdout.strip())


def _encode_m4b_with_chapters(input_audio: Path, outfile: Path, metadata_file: Path, bitrate: str, threads: int) -> None:
    use_fdk = False
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if encoders.returncode == 0 and "libfdk_aac" in encoders.stdout:
        use_fdk = True

    aac_encoder = "libfdk_aac" if use_fdk else "aac"
    command = [
        "ffmpeg",
        "-y",
        "-threads",
        str(threads),
        "-i",
        str(input_audio),
        "-i",
        str(metadata_file),
        "-map_metadata",
        "1",
        "-map_chapters",
        "1",
        "-vn",
        "-c:a",
        aac_encoder,
        "-b:a",
        bitrate,
        "-movflags",
        "+faststart",
        str(outfile),
    ]

    cp = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed:\n{cp.stderr}")


def _seconds_to_timestamp_ms(seconds: float) -> str:
    return str(int(round(seconds * 1000)))


def _write_ffmetadata(chapters: list[tuple[float, float, str]], out_path: Path) -> None:
    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as fh:
        tmp_path = Path(fh.name)
        fh.write(";FFMETADATA1\n")
        for start, end, title in chapters:
            safe_title = title.replace(";", r"\;").replace("=", r"\=")
            fh.write("[CHAPTER]\n")
            fh.write("TIMEBASE=1/1000\n")
            fh.write(f"START={_seconds_to_timestamp_ms(start)}\n")
            fh.write(f"END={_seconds_to_timestamp_ms(end)}\n")
            fh.write(f"title={safe_title}\n")
    tmp_path.replace(out_path)


def _require_binary(binary_name: str) -> None:
    if shutil.which(binary_name) is None:
        raise RuntimeError(f"Required binary not found in PATH: {binary_name}")


if __name__ == "__main__":
    script()
