"""audio_prep.py

Offline audio preprocessing pipeline for the Music Visualizer project.

This module is responsible for turning raw .wav files into two kinds of
artifacts that the Swift/Metal renderer can consume:

1. **Per–song feature files** (compressed .npz)
   - Produced by :func:`extractSongFeatures` via :func:`getSongFeatures`.
   - These files store STFT magnitudes, loudness metrics, histograms, and
     other quantities that are useful for both Song Mode and Trend Mode.

2. **Per–song point clouds** (raw .bin)
   - Produced by :func:`exportSongPointCloud`.
   - Each vertex encodes a single time–frequency sample with the following
     layout per vertex (float32, interleaved):

         [x, y, z, scalar, hp_ratio]

       where
         - x: normalized time in [-1, 1]
         - y: log amplitude (dB-like) normalized to [-1, 1]
         - z: normalized frequency index in [-1, 1]
         - scalar: perceived loudness (A-weighted), normalized to [0, 1]
         - hp_ratio: harmonic–percussive ratio in [0, 1]

   - These are consumed directly by the GPU for Song Mode rendering.

3. **Per–decade trend feature files** (compressed .npz)
   - Produced by :func:`aggregateTrendFeatures` after all songs in a decade
     have been processed.
   - These capture aggregated statistics (e.g., mean amplitude histogram
     and per-song loudness/dynamic range) that are later used to construct
     the decade-level scalar field for Trend Mode.

Directory layout (relative to the project root):

    ./data/music/<decade>/some_song.wav
    ./data/processed/<decade>_songs/<song_stem>_features.npz
    ./data/processed/<decade>_songs/<song_stem>_points.bin
    ./data/processed/<decade>_trend_features.npz

High-level flow:

1. :func:`main` calls :func:`extractSongFeatures`.
2. :func:`extractSongFeatures` walks all decades under ``data/music`` and,
   for each .wav file:
      a. Calls :func:`getSongFeatures` to compute spectrogram + loudness
         + histogram features.
      b. Saves the feature dict to an ``*_features.npz`` file.
      c. Calls :func:`exportSongPointCloud` to create the dense per-song
         point cloud used in Song Mode.
3. Once all songs in a decade are processed, :func:`aggregateTrendFeatures`
   is called to produce a single ``<decade>_trend_features.npz`` file.

This module is purely offline preprocessing and does not depend on Metal. It
is intended to be run occasionally when the dataset changes or when feature
definitions are updated.
"""

from pathlib import Path
import numpy as np

try:  # librosa is only required for feature extraction, not for timeline stitching.
    import librosa
except ImportError:  # pragma: no cover - optional dependency for offline runs
    librosa = None


def aggregateTrendFeatures(decadePath: str) -> None:
    """Aggregate per-song trend features in a processed decade directory.

    Parameters
    ----------
    decadePath : str
        Path to a processed decade directory, e.g.::

            ./data/processed/1980s_songs

    This function expects that directory to contain per-song
    ``*_features.npz`` files produced by :func:`extractSongFeatures`.

    For each compatible file, it loads:

      - ``amp_hist``: histogram of absolute sample amplitudes for that song.
      - ``amp_bin_edges``: the shared bin edges for the amplitude histogram.
      - ``global_loudness_db`` and ``dynamic_range_db``: simple loudness
        metrics computed from frame-wise RMS.

    It then aggregates these across all songs in the decade to produce
    decade-level statistics and writes a single compressed file::

        ./data/processed/<decade>_trend_features.npz

    Saved decade-level fields
    -------------------------
    decade : str
        Label of the decade (e.g. "1980s").
    amp_bin_edges : ndarray, shape [n_amp_bins + 1]
        Shared amplitude histogram bin edges.
    mean_amp_hist : ndarray, shape [n_amp_bins]
        Mean of the amplitude histograms over all songs in the decade.
    std_amp_hist : ndarray, shape [n_amp_bins]
        Standard deviation of the amplitude histograms over songs.
    per_song_global_loudness_db : ndarray, shape [num_songs]
        Global loudness (mean dB over frames) for each song.
    per_song_dynamic_range_db : ndarray, shape [num_songs]
        Dynamic range (95th - 5th percentile of loudness dB) per song.
    song_files : ndarray, shape [num_songs]
        Filenames of the contributing per-song feature files.
    """
    decade_dir = Path(decadePath)

    if not decade_dir.exists():
        print(f"[aggregateTrendFeatures] Processed decade dir not found: {decade_dir}")
        return

    # Collect per-song statistics across the decade.
    amp_hists: list[np.ndarray] = []
    global_loudness_list: list[float] = []
    dynamic_range_list: list[float] = []
    song_files: list[str] = []
    amp_bin_edges: np.ndarray | None = None

    # Iterate over all .npz files in the decade directory. Older or partial
    # files that do not contain the expected fields are skipped.
    for npz_path in sorted(decade_dir.glob("*.npz")):
        try:
            with np.load(npz_path) as data:
                if "amp_hist" not in data or "amp_bin_edges" not in data:
                    # Older/partial feature files; skip them.
                    print(
                        f"[aggregateTrendFeatures] Skipping {npz_path.name} "
                        "(missing amp_hist / amp_bin_edges)."
                    )
                    continue

                amp_hist = data["amp_hist"]
                amp_hists.append(amp_hist)

                # All songs in a decade share the same bin edges by design.
                if amp_bin_edges is None:
                    amp_bin_edges = data["amp_bin_edges"]

                # Loudness metrics are optional but recommended; if present,
                # accumulate them for later decade-level statistics.
                if "global_loudness_db" in data:
                    global_loudness_list.append(float(data["global_loudness_db"]))
                if "dynamic_range_db" in data:
                    dynamic_range_list.append(float(data["dynamic_range_db"]))

                song_files.append(npz_path.name)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[aggregateTrendFeatures] Error loading {npz_path}: {exc}")

    if not amp_hists:
        print(
            f"[aggregateTrendFeatures] No compatible song feature files "
            f"found in {decade_dir}"
        )
        return

    # Stack histograms into a 2D array: (num_songs, n_bins)
    amp_hists_arr = np.stack(amp_hists, axis=0)
    mean_amp_hist = amp_hists_arr.mean(axis=0)
    std_amp_hist = amp_hists_arr.std(axis=0)

    # Convert optional lists to arrays (may be empty if missing).
    global_loudness_arr = (
        np.array(global_loudness_list, dtype=np.float32)
        if global_loudness_list
        else np.array([], dtype=np.float32)
    )
    dynamic_range_arr = (
        np.array(dynamic_range_list, dtype=np.float32)
        if dynamic_range_list
        else np.array([], dtype=np.float32)
    )

    # Derive a label like "1980s" from the directory name ("1980s_songs").
    decade_label = decade_dir.name.replace("_songs", "")
    out_path = decade_dir.parent / f"{decade_label}_trend_features.npz"

    # Persist the aggregated statistics to disk.
    np.savez_compressed(
        out_path,
        decade=np.array(decade_label),
        amp_bin_edges=amp_bin_edges.astype(np.float32),
        mean_amp_hist=mean_amp_hist.astype(np.float32),
        std_amp_hist=std_amp_hist.astype(np.float32),
        per_song_global_loudness_db=global_loudness_arr,
        per_song_dynamic_range_db=dynamic_range_arr,
        song_files=np.array(song_files),
    )

    print(f"[aggregateTrendFeatures] Wrote trend features for {decade_label} -> {out_path}")


def getSongFeatures(songPath: str) -> dict:
    """Compute per-song audio features for Song Mode and Trend Mode.

    This is the core feature-extraction routine. It takes a path to a
    single .wav file, computes a variety of time–frequency and loudness
    descriptors, and returns them in a dictionary that can be written
    directly with :func:`numpy.savez_compressed`.

    Parameters
    ----------
    songPath : str
        Path to a .wav file.

    Returns
    -------
    dict
        Dictionary of NumPy arrays / scalars ready to be saved.

    Returned fields
    ---------------
    sample_rate : int
        Native sample rate of the audio file.
    n_fft : int
        FFT window size used for the STFT.
    hop_length : int
        Hop length between consecutive STFT frames.
    magnitude : ndarray, shape [freq_bins, frames]
        Magnitude spectrogram |STFT|.
    hp_ratio : ndarray, shape [freq_bins, frames]
        Harmonic–percussive energy ratio per time–frequency bin. Values in
        [0, 1], where 1 indicates fully harmonic energy and 0 indicates
        fully percussive.
    rms : ndarray, shape [frames]
        Frame-wise RMS of the time-domain signal.
    magnitude_db : ndarray, shape [freq_bins, frames]
        Magnitude spectrogram converted to decibels using
        :func:`librosa.amplitude_to_db` with ``ref=np.max``.
    loudness_db : ndarray, shape [frames]
        A simple loudness approximation in dB derived from RMS.
    loudness_A_db : ndarray, shape [frames]
        A-weighted loudness per frame, capturing a perceptual weighting
        of frequencies.
    global_loudness_A_db : float
        Mean A-weighted loudness over all frames.
    dynamic_range_A_db : float
        Dynamic range of A-weighted loudness (95th - 5th percentile).
    amp_hist : ndarray, shape [amp_bins]
        Histogram of absolute sample amplitudes in the time domain. Used
        for building decade-level amplitude distributions in Trend Mode.
    amp_bin_edges : ndarray, shape [amp_bins + 1]
        Bin edges corresponding to ``amp_hist``.
    global_loudness_db : float
        Mean loudness (non A-weighted) over frames.
    dynamic_range_db : float
        Dynamic range of non A-weighted loudness (95th - 5th percentile).
    """
    if librosa is None:
        raise ImportError("librosa is required for getSongFeatures; please install it.")

    song_path = Path(songPath)

    # ------------------------------------------------------------------
    # 1. Load audio (mono) and keep the native sample rate.
    # ------------------------------------------------------------------
    # librosa.load returns float32 samples in [-1, 1].
    y, sr = librosa.load(song_path, sr=None, mono=True)

    # ------------------------------------------------------------------
    # 2. Compute STFT and magnitude spectrogram.
    # ------------------------------------------------------------------
    # These parameters should match whatever the renderer assumes when
    # interpreting the per-song spectrogram.
    n_fft = 2048
    hop_length = 1024

    stft = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann",
        center=True,
    )
    magnitude = np.abs(stft).astype(np.float32)  # (freq_bins, frames)

    # ------------------------------------------------------------------
    # 3. Harmonic–percussive source separation for hue information.
    # ------------------------------------------------------------------
    # ``hp_ratio`` encodes the balance between harmonic and percussive
    # energy in each time–frequency bin and is later mapped to hue.
    H, P = librosa.decompose.hpss(stft)
    harmonic_mag = np.abs(H).astype(np.float32)
    percussive_mag = np.abs(P).astype(np.float32)
    hp_eps = 1e-10
    hp_ratio = harmonic_mag / (harmonic_mag + percussive_mag + hp_eps)

    # ------------------------------------------------------------------
    # 4. Convert magnitude to dB for visualization / loudness-like values.
    # ------------------------------------------------------------------
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max).astype(np.float32)

    # ------------------------------------------------------------------
    # 5. Compute A-weighted loudness per frame.
    # ------------------------------------------------------------------
    # A-weighting emphasizes frequencies human hearing is most sensitive to.
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    freqs_safe = np.maximum(freqs, 1.0)  # avoid log(0) in A-weighting
    A_weight_db = librosa.A_weighting(freqs_safe)
    A_weight_gain = 10.0 ** (A_weight_db / 20.0)

    # Apply A-weighting in the frequency domain and then reduce to a
    # single loudness value per frame via RMS.
    magnitude_A = magnitude * A_weight_gain[:, np.newaxis]
    eps = 1e-10
    aw_rms = np.sqrt(np.mean(magnitude_A ** 2, axis=0) + eps).astype(np.float32)
    loudness_A_db = (20.0 * np.log10(aw_rms + eps)).astype(np.float32)

    global_loudness_A_db = float(loudness_A_db.mean())
    p95_A = float(np.percentile(loudness_A_db, 95))
    p5_A = float(np.percentile(loudness_A_db, 5))
    dynamic_range_A_db = p95_A - p5_A

    # ------------------------------------------------------------------
    # 6. Frame-wise RMS and non A-weighted loudness.
    # ------------------------------------------------------------------
    rms = librosa.feature.rms(
        y=y,
        frame_length=n_fft,
        hop_length=hop_length,
        center=True,
    )[0].astype(np.float32)

    loudness_db = (20.0 * np.log10(rms + eps)).astype(np.float32)
    global_loudness_db = float(loudness_db.mean())

    p95 = float(np.percentile(loudness_db, 95))
    p5 = float(np.percentile(loudness_db, 5))
    dynamic_range_db = p95 - p5

    # ------------------------------------------------------------------
    # 7. Amplitude histogram of the raw waveform (Trend Mode feature).
    # ------------------------------------------------------------------
    # ``amp`` captures the full distribution of instantaneous amplitudes in
    # the time domain, which is used later to build decade-level summary
    # histograms in Trend Mode.
    amp = np.abs(y)
    n_amp_bins = 64
    amp_hist, amp_bin_edges = np.histogram(amp, bins=n_amp_bins, range=(0.0, 1.0))

    return {
        "sample_rate": np.int32(sr),
        "n_fft": np.int32(n_fft),
        "hop_length": np.int32(hop_length),
        "magnitude": magnitude,
        "hp_ratio": hp_ratio,
        "rms": rms,
        "magnitude_db": magnitude_db,
        "loudness_db": loudness_db,
        "loudness_A_db": loudness_A_db,
        "amp_hist": amp_hist.astype(np.int32),
        "amp_bin_edges": amp_bin_edges.astype(np.float32),
        "global_loudness_db": np.float32(global_loudness_db),
        "dynamic_range_db": np.float32(dynamic_range_db),
        "global_loudness_A_db": np.float32(global_loudness_A_db),
        "dynamic_range_A_db": np.float32(dynamic_range_A_db),
    }


def exportSongPointCloud(features: dict, outPath: str) -> None:
    """Export a song-mode point cloud for visualization.

    This function converts per-song STFT/loudness features into a dense
    3D point cloud that can be rendered directly by the GPU. The point
    cloud encodes a time–frequency volume and associated scalar fields.

    The point layout per vertex is::

        [x, y, z, scalar, hp_ratio]

    where
        x : float
            Normalized time in [-1, 1].
        y : float
            Log amplitude (dB-like) mapped to [-1, 1].
        z : float
            Normalized frequency index in [-1, 1].
        scalar : float
            Perceived loudness (A-weighted) mapped to [0, 1]. This is the
            scalar field that the isosurface extraction operates on.
        hp_ratio : float
            Harmonic–percussive ratio in [0, 1], used as a hue dimension.

    The output is a raw float32 binary file written with :meth:`ndarray.tofile`.

    Notes
    -----
    * Time frames that are completely silent (no energy across frequencies)
      are detected and removed to avoid a large degenerate slab in the
      point cloud at y = -1.
    * All normalizations (for amplitude and loudness) are performed over
      the remaining, non-silent points.
    """
    if librosa is None:
        raise ImportError("librosa is required for exportSongPointCloud; please install it.")

    required_keys = ("magnitude", "hp_ratio", "sample_rate", "n_fft")
    if not all(k in features for k in required_keys):
        print(
            "[exportSongPointCloud] Missing required feature keys in song "
            f"features; expected {required_keys}. Skipping point cloud export."
        )
        return

    magnitude = features["magnitude"]        # [freq_bins, frames]
    hp_ratio = features["hp_ratio"]          # [freq_bins, frames]
    sample_rate = int(features["sample_rate"])
    n_fft = int(features["n_fft"])

    freq_bins, frames = magnitude.shape

    # Guard against degenerate input.
    if freq_bins <= 0 or frames <= 0:
        print("[exportSongPointCloud] Empty magnitude array; skipping.")
        return

    # Frequency and time indices used for building the grid.
    freqs_idx = np.arange(freq_bins, dtype=int)
    times_idx = np.arange(frames, dtype=int)

    # Amplitude per (freq, time) on the full grid.
    amp = magnitude[np.ix_(freqs_idx, times_idx)]  # [F, T]

    # ------------------------------------------------------------------
    # 1. Identify and drop completely silent time frames.
    # ------------------------------------------------------------------
    # A frame is considered silent if every frequency bin has zero energy.
    col_max = amp.max(axis=0)  # [T]
    valid_mask = col_max > 0.0
    if not np.any(valid_mask):
        print("[exportSongPointCloud] All frames are silent; skipping.")
        return

    # Keep only non-silent time frames.
    times_idx = times_idx[valid_mask]
    amp = amp[:, valid_mask]

    # Recompute frame count after filtering.
    frames_valid = amp.shape[1]

    # ------------------------------------------------------------------
    # 2. Compute log amplitude (dB-like) for better dynamic range.
    # ------------------------------------------------------------------
    amp_eps = 1e-10
    amp_log = 20.0 * np.log10(amp + amp_eps)  # typically negative values
    # Floor extremely quiet bins to avoid a hard clamp after normalization.
    # Using a low percentile prevents a single near-zero bin from setting the global min.
    amp_floor = np.percentile(amp_log, 5.0)
    amp_log = np.maximum(amp_log, amp_floor)

    # ------------------------------------------------------------------
    # 3. Compute per-bin A-weighted loudness as the scalar field.
    # ------------------------------------------------------------------
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    freqs_safe = np.maximum(freqs, 1.0)
    A_weight_db = librosa.A_weighting(freqs_safe)
    A_weight_gain = 10.0 ** (A_weight_db / 20.0)

    # Apply A-weighting to each frequency bin, then collapse to dB.
    amp_A = amp * A_weight_gain[:, np.newaxis]
    loudness_A_db = 20.0 * np.log10(amp_A + amp_eps)

    # ------------------------------------------------------------------
    # 4. Normalize log amplitude and loudness for stable rendering.
    # ------------------------------------------------------------------
    # Normalize log amplitude globally over remaining points to [0, 1].
    log_min = float(amp_log.min())
    log_max = float(amp_log.max())
    if log_max == log_min:
        amp_norm = np.zeros_like(amp_log, dtype=np.float32)
    else:
        amp_norm = ((amp_log - log_min) / (log_max - log_min)).astype(np.float32)

    # Normalize per-bin A-weighted loudness globally to [0, 1].
    loud_min = float(loudness_A_db.min())
    loud_max = float(loudness_A_db.max())
    if loud_max == loud_min:
        loudness_norm = np.zeros_like(loudness_A_db, dtype=np.float32)
    else:
        loudness_norm = (
            (loudness_A_db - loud_min) / (loud_max - loud_min)
        ).astype(np.float32)

    # Map amplitude to [-1, 1] for the y-axis in NDC space.
    amp_ndc = amp_norm * 2.0 - 1.0

    # ------------------------------------------------------------------
    # 5. Build normalized coordinate grid for time (x) and frequency (z).
    # ------------------------------------------------------------------
    # Normalized time (0..1) based on remaining (non-silent) frames, then
    # mapped to [-1, 1]. This preserves temporal ordering but compresses
    # any gaps produced by dropping silent frames.
    t_norm = times_idx / max(times_idx[-1], 1)
    t_ndc = t_norm * 2.0 - 1.0  # [T]

    # Normalized frequency index (0..1) then to [-1, 1].
    f_norm = freqs_idx / max(freq_bins - 1, 1)
    f_ndc = f_norm * 2.0 - 1.0  # [F]

    # Broadcast positions: X = time, Y = log amplitude, Z = frequency.
    X = np.broadcast_to(t_ndc, (freq_bins, frames_valid))
    Z = np.broadcast_to(f_ndc[:, None], (freq_bins, frames_valid))
    Y = amp_ndc

    # Sample harmonic–percussive ratio on the same (freq, time) grid.
    hp = hp_ratio[np.ix_(freqs_idx, times_idx)]  # [F, frames_valid]

    # ------------------------------------------------------------------
    # 6. Flatten into [N, 5] float32 and write to disk.
    # ------------------------------------------------------------------
    X_flat = X.ravel().astype(np.float32)
    Y_flat = Y.ravel().astype(np.float32)
    Z_flat = Z.ravel().astype(np.float32)
    loud_flat = loudness_norm.ravel().astype(np.float32)
    hp_flat = hp.ravel().astype(np.float32)

    verts = np.stack(
        [X_flat, Y_flat, Z_flat, loud_flat, hp_flat],
        axis=1,
    ).astype(np.float32)

    out_path = Path(outPath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    verts.tofile(out_path)

    print(
        f"[exportSongPointCloud] Wrote point cloud: {out_path} "
        f"(vertices: {verts.shape[0]})"
    )


def buildTimelinePointCloud(
    processedRoot: str | Path = "data/processed",
    outPath: str | Path | None = None,
    maxFramesPerSong: int = 50000,
    songTimeScale: float = 0.8,
    baseGap: float = 0.02,
    yearGapScale: float = 0.005,
    separatorCount: int = 24,
    separatorScalar: float = 0.2,
    crestFloorDb: float = 0.0,
    crestCeilDb: float = 30.0,
) -> None:
    """Construct a single point cloud that stitches all songs chronologically.

    Each point encodes a time sample from a song with the layout::

        [x, y, z, scalar, hp_ratio]

    Axis meanings for the stitched timeline:
        * x: in-song time scaled by ``songTimeScale`` plus a translation that
             accumulates with year gaps so songs appear in chronological order.
        * y: normalized amplitude (RMS) mapped to [-1, 1].
        * z: per-frame crest factor (peak vs RMS) mapped to [-1, 1].
        * scalar: normalized A-weighted loudness, used by the renderer for
                  alpha/intensity.
        * hp_ratio: normalized release year in [0, 1] to give a color ramp
                    across the timeline.

    Parameters
    ----------
    processedRoot : str | Path
        Root directory containing ``*_songs`` folders with ``*_features.npz``.
    outPath : str | Path | None
        Destination for the stitched binary. Defaults to
        ``<processedRoot>/timeline_points.bin``.
    maxFramesPerSong : int
        Downsamples each song to at most this many frames to keep memory usage
        reasonable.
    songTimeScale : float
        Multiplier applied to normalized song time. Reduce to shrink the total
        X-span; increase to stretch.
    baseGap : float
        Fixed gap between consecutive songs along X (kept small for a continuous ribbon).
    yearGapScale : float
        Additional gap per year difference between consecutive songs.
    separatorCount : int
        Number of points to emit for a vertical separator between songs.
    separatorScalar : float
        Scalar (alpha driver) for separator points; keep small so they read subtly.
    crestFloorDb : float
        Minimum crest factor (dB) used for normalization; values below are clamped.
    crestCeilDb : float
        Maximum crest factor (dB) used for normalization; values above are clamped.
    """
    root = Path(processedRoot)
    if outPath is None:
        outPath = root / "timeline_points.bin"
    else:
        outPath = Path(outPath)

    song_feature_paths: list[Path] = []
    for decade_dir in sorted(root.glob("*_songs")):
        if decade_dir.is_dir():
            song_feature_paths.extend(sorted(decade_dir.glob("*_features.npz")))

    if not song_feature_paths:
        print(f"[buildTimelinePointCloud] No feature files found under {root}")
        return

    songs: list[dict] = []
    for feat_path in song_feature_paths:
        # Extract a 4-digit year prefix from the filename.
        year = None
        stem = feat_path.name
        if len(stem) >= 4 and stem[:4].isdigit():
            year = int(stem[:4])
        if year is None:
            print(f"[buildTimelinePointCloud] Skipping (no year prefix): {feat_path}")
            continue

        try:
            with np.load(feat_path) as data:
                if (
                    "rms" not in data
                    or "hop_length" not in data
                    or "sample_rate" not in data
                    or "magnitude_db" not in data
                    or "loudness_db" not in data
                    or "hp_ratio" not in data
                ):
                    print(
                        f"[buildTimelinePointCloud] Missing required fields in {feat_path.name};"
                        " expected rms, magnitude_db, loudness_db, hp_ratio, hop_length, sample_rate."
                    )
                    continue
                rms = np.asarray(data["rms"], dtype=np.float32)
                magnitude_db = np.asarray(data["magnitude_db"], dtype=np.float32)
                loudness_db = np.asarray(data["loudness_db"], dtype=np.float32)
                hp_ratio = np.asarray(data["hp_ratio"], dtype=np.float32)
                loud_A = (
                    np.asarray(data["loudness_A_db"], dtype=np.float32)
                    if "loudness_A_db" in data
                    else None
                )
                hop_length = int(data["hop_length"])
                sample_rate = int(data["sample_rate"])
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[buildTimelinePointCloud] Error reading {feat_path.name}: {exc}")
            continue

        if rms.size == 0:
            continue

        frames = rms.shape[0]
        duration_sec = float(frames * hop_length) / max(sample_rate, 1)
        time_sec = np.arange(frames, dtype=np.float32) * (hop_length / float(sample_rate))
        peak_db = magnitude_db.max(axis=0)
        crest_db = np.clip(peak_db - loudness_db, crestFloorDb, crestCeilDb).astype(np.float32)
        frame_hp = hp_ratio.mean(axis=0)
        if frame_hp.shape[0] > frames:
            frame_hp = frame_hp[:frames]
        elif frame_hp.shape[0] < frames:
            frame_hp = np.pad(frame_hp, (0, frames - frame_hp.shape[0]), mode="edge")
        frame_hp = np.clip(frame_hp, 0.0, 1.0).astype(np.float32)

        songs.append(
            {
                "name": feat_path.name,
                "year": year,
                "rms": rms,
                "loud_array": loud_A
                if loud_A is not None
                else (20.0 * np.log10(np.maximum(rms, 1e-10))),
                "hop_length": hop_length,
                "sample_rate": sample_rate,
                "duration_sec": duration_sec,
                "time_sec": time_sec,
                "crest_array": crest_db,
                "hp_frame": frame_hp,
            }
        )

    if not songs:
        print("[buildTimelinePointCloud] No songs with usable features.")
        return

    # Global ranges for normalization.
    min_rms = min(float(np.min(s["rms"])) for s in songs)
    max_rms = max(float(np.max(s["rms"])) for s in songs)
    min_dyn = crestFloorDb
    max_dyn = crestCeilDb
    min_year = min(s["year"] for s in songs)
    max_year = max(s["year"] for s in songs)

    # Loudness range uses A-weighted dB if present; otherwise falls back to log RMS.
    min_loud = min(float(np.min(s["loud_array"])) for s in songs)
    max_loud = max(float(np.max(s["loud_array"])) for s in songs)

    max_duration = max(s["duration_sec"] for s in songs)
    duration_scale = songTimeScale / max(max_duration, 1e-6)

    rms_range = max(max_rms - min_rms, 1e-6)
    loud_range = max(max_loud - min_loud, 1e-6)
    dyn_range = max(max_dyn - min_dyn, 1e-6)
    year_range = max(max_year - min_year, 1)

    songs_sorted = sorted(songs, key=lambda s: (s["year"], s["name"]))
    verts: list[np.ndarray] = []

    offset_x = 0.0
    prev_year = None
    prev_span = 0.0

    for idx, song in enumerate(songs_sorted):
        year = song["year"]
        if prev_year is not None:
            year_gap = max(year - prev_year, 0)
            offset_x += prev_span + baseGap + year_gap * yearGapScale
        prev_year = year

        # Downsample frames.
        frames = song["rms"].shape[0]
        if frames > maxFramesPerSong:
            sample_idx = np.linspace(0, frames - 1, maxFramesPerSong, dtype=int)
        else:
            sample_idx = np.arange(frames, dtype=int)

        t_scaled = song["time_sec"][sample_idx] * duration_scale
        span = t_scaled.max() - t_scaled.min() if t_scaled.size > 0 else 0.0
        prev_span = span

        rms = song["rms"][sample_idx]
        rms01 = (rms - min_rms) / rms_range
        amp_ndc = rms01 * 2.0 - 1.0

        loud_arr = song["loud_array"][sample_idx]
        loud01 = (loud_arr - min_loud) / loud_range

        dyn = song["crest_array"][sample_idx]
        dyn01 = (dyn - min_dyn) / dyn_range
        dyn_ndc = dyn01 * 2.0 - 1.0

        year01 = (year - min_year) / year_range

        x = offset_x + t_scaled
        y = amp_ndc.astype(np.float32)
        z = dyn_ndc.astype(np.float32)
        scalar = loud01.astype(np.float32)
        hp = song["hp_frame"][sample_idx].astype(np.float32)

        verts.append(
            np.stack(
                [
                    x.astype(np.float32),
                    y,
                    z,
                    scalar,
                    hp,
                ],
                axis=1,
            )
        )

        # Emit a slim vertical separator between this song and the next.
        if idx < len(songs_sorted) - 1 and separatorCount > 0:
            next_year = songs_sorted[idx + 1]["year"]
            sep_gap = baseGap + max(next_year - year, 0) * yearGapScale
            sep_x = (offset_x + span) + sep_gap * 0.5
            y_sep = np.linspace(-1.0, 1.0, separatorCount, dtype=np.float32)
            dyn_sep = float(np.median(dyn_ndc)) if dyn_ndc.size > 0 else 0.0
            hp_vals = song["hp_frame"][sample_idx]
            hp_sep = float(np.median(hp_vals)) if hp_vals.size > 0 else 0.0
            verts.append(
                np.stack(
                    [
                        np.full_like(y_sep, np.float32(sep_x)),
                        y_sep,
                        np.full_like(y_sep, np.float32(dyn_sep)),
                        np.full_like(y_sep, np.float32(separatorScalar)),
                        np.full_like(y_sep, np.float32(hp_sep)),
                    ],
                    axis=1,
                )
            )

    if not verts:
        print("[buildTimelinePointCloud] No vertices generated.")
        return

    stitched = np.concatenate(verts, axis=0)
    outPath.parent.mkdir(parents=True, exist_ok=True)
    stitched.astype(np.float32).tofile(outPath)
    print(
        f"[buildTimelinePointCloud] Wrote timeline cloud: {outPath} "
        f"(songs: {len(songs_sorted)}, vertices: {stitched.shape[0]})"
    )


def extractSongFeatures() -> None:
    """Walk ``./data/music`` and extract per-song features.

    For every decade directory under ``./data/music``, this function:

    1. Discovers all ``.wav``/``.wave`` files.
    2. For each file, calls :func:`getSongFeatures` to compute features.
    3. Saves the resulting dictionary into a compressed ``*_features.npz``
       file under ``./data/processed/<decade>_songs``.
    4. Calls :func:`exportSongPointCloud` to write a dense point cloud
       ``*_points.bin`` for Song Mode visualization.
    5. After all songs in a decade are processed, calls
       :func:`aggregateTrendFeatures` to compute a single
       ``<decade>_trend_features.npz`` file for Trend Mode.

    The function is idempotent with respect to directory layout: if the
    processed directories already exist, they are reused and new files are
    simply written/overwritten as needed.
    """
    project_root = Path(__file__).resolve().parent
    music_root = project_root / "data" / "music"
    processed_root = project_root / "data" / "processed"

    if not music_root.exists():
        print(f"[extractSongFeatures] Music root not found: {music_root}")
        return

    # Ensure the global processed root exists.
    processed_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Iterate over each decade directory under ./data/music.
    # ------------------------------------------------------------------
    for decade_dir in sorted(music_root.iterdir()):
        if not decade_dir.is_dir():
            continue

        decade_label = decade_dir.name
        print(f"[extractSongFeatures] Processing decade: {decade_label}")

        # Output directory for this decade's song-level features.
        decade_out_dir = processed_root / f"{decade_label}_songs"
        decade_out_dir.mkdir(parents=True, exist_ok=True)

        # Collect all supported audio files in this decade.
        song_files = sorted(
            p
            for p in decade_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".wav", ".wave"}
        )

        if not song_files:
            print(f"[extractSongFeatures] No .wav files found in {decade_dir}")
            continue

        # ------------------------------------------------------------------
        # Per-song processing loop.
        # ------------------------------------------------------------------
        for song_path in song_files:
            print(f"[extractSongFeatures]   {song_path.name}")
            try:
                features = getSongFeatures(str(song_path))
            except Exception as exc:  # pragma: no cover - defensive logging
                print(
                    f"[extractSongFeatures]   ERROR processing "
                    f"{song_path.name}: {exc}"
                )
                continue

            # Construct output paths for the feature NPZ and point cloud BIN.
            out_name = f"{song_path.stem}_features.npz"
            out_path = decade_out_dir / out_name
            bin_name = f"{song_path.stem}_points.bin"
            bin_path = decade_out_dir / bin_name

            # Save features plus metadata about decade and song filename.
            np.savez_compressed(
                out_path,
                decade=np.array(decade_label),
                song=np.array(song_path.name),
                **features,
            )

            rel_out = out_path.relative_to(project_root)
            rel_bin_out = bin_path.relative_to(project_root)
            print(f"[extractSongFeatures]   -> {rel_out}")

            # Export the dense point cloud used for Song Mode visualization.
            exportSongPointCloud(features, rel_bin_out)

        # After processing all songs for this decade, aggregate trend features.
        aggregateTrendFeatures(str(decade_out_dir))

    # Build a stitched all-songs timeline once per full extraction run.
    buildTimelinePointCloud(processedRoot=processed_root)


def rebuildTimeline(processedRoot: str | Path = "data/processed",
                    outPath: str | Path | None = None) -> None:
    """Convenience wrapper to rebuild only the stitched timeline cloud."""
    buildTimelinePointCloud(processedRoot=processedRoot, outPath=outPath)


def main() -> None:
    """Entry point for the audio preprocessing pipeline.

    This simply invokes :func:`extractSongFeatures`. Keeping it as a
    separate function makes it easier to call from other scripts or tests.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Audio preprocessing pipeline.")
    parser.add_argument(
        "--timeline-only",
        action="store_true",
        help="Only rebuild the stitched timeline point cloud.",
    )
    parser.add_argument(
        "--processed-root",
        default="data/processed",
        help="Root folder containing *_songs feature folders (default: data/processed).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output path for timeline_points.bin (default under processed root).",
    )
    args = parser.parse_args()

    if args.timeline_only:
        rebuildTimeline(processedRoot=args.processed_root, outPath=args.out)
    else:
        extractSongFeatures()


if __name__ == "__main__":
    main()
