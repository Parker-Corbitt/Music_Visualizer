from pathlib import Path
import numpy as np
import librosa

def aggregateTrendFeatures(decadePath: str) -> None:
    """Aggregate per-song trend features in a processed decade directory.

    Parameters
    ----------
    decadePath : str
        Path to a processed decade directory
            ./data/processed/1980s_songs

    This function:
        - Loads all per-song *_features.npz files in the decade directory
        - Aggregates amplitude histograms and loudness metrics
        - Writes a decade-level trend feature file:
              ./data/processed/<decade>_trend_features.npz

    Saved decade-level fields:
        - decade: str
        - amp_bin_edges: 1D ndarray [n_amp_bins + 1]
        - mean_amp_hist: 1D ndarray [n_amp_bins]
        - std_amp_hist: 1D ndarray [n_amp_bins]
        - per_song_global_loudness_db: 1D ndarray [num_songs]
        - per_song_dynamic_range_db: 1D ndarray [num_songs]
        - song_files: 1D ndarray [num_songs] of filenames
    """
    decade_dir = Path(decadePath)

    if not decade_dir.exists():
        print(f"[aggregateTrendFeatures] Processed decade dir not found: {decade_dir}")
        return

    amp_hists = []
    global_loudness_list = []
    dynamic_range_list = []
    song_files: list[str] = []
    amp_bin_edges = None

    for npz_path in sorted(decade_dir.glob("*.npz")):
        try:
            with np.load(npz_path) as data:
                if "amp_hist" not in data or "amp_bin_edges" not in data:
                    # Older/partial feature files; skip them.
                    print(f"[aggregateTrendFeatures] Skipping {npz_path.name} (missing amp_hist / amp_bin_edges).")
                    continue

                amp_hist = data["amp_hist"]
                amp_hists.append(amp_hist)

                if amp_bin_edges is None:
                    amp_bin_edges = data["amp_bin_edges"]

                if "global_loudness_db" in data:
                    global_loudness_list.append(float(data["global_loudness_db"]))
                if "dynamic_range_db" in data:
                    dynamic_range_list.append(float(data["dynamic_range_db"]))

                song_files.append(npz_path.name)
        except Exception as exc:
            print(f"[aggregateTrendFeatures] Error loading {npz_path}: {exc}")

    if not amp_hists:
        print(f"[aggregateTrendFeatures] No compatible song feature files found in {decade_dir}")
        return

    amp_hists_arr = np.stack(amp_hists, axis=0)  # (num_songs, n_bins)
    mean_amp_hist = amp_hists_arr.mean(axis=0)
    std_amp_hist = amp_hists_arr.std(axis=0)

    global_loudness_arr = np.array(global_loudness_list, dtype=np.float32) if global_loudness_list else np.array([], dtype=np.float32)
    dynamic_range_arr = np.array(dynamic_range_list, dtype=np.float32) if dynamic_range_list else np.array([], dtype=np.float32)

    decade_label = decade_dir.name.replace("_songs", "")
    out_path = decade_dir.parent / f"{decade_label}_trend_features.npz"

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
    """Compute per-song audio features for Song/Trend modes.

    Parameters
    ----------
    songPath : str
        Path to a .wav file.

    Returns
    -------
    dict
        Dictionary of NumPy arrays / scalars ready to be saved via np.savez_compressed.

    Fields:
        - sample_rate: int
        - n_fft: int
        - hop_length: int
        - magnitude: 2D ndarray [freq_bins, frames]
        - hp_ratio: 2D ndarray [freq_bins, frames] (harmonic–percussive energy ratio per time–frequency bin)
        - rms: 1D ndarray [frames]
        - magnitude_db: 2D ndarray [freq_bins, frames]
        - loudness_db: 1D ndarray [frames]
        - loudness_A_db: 1D ndarray [frames] (A-weighted loudness per frame)
        - global_loudness_A_db: float (mean A-weighted loudness)
        - dynamic_range_A_db: float (A-weighted dynamic range)
        - amp_hist: 1D ndarray [amp_bins]
        - amp_bin_edges: 1D ndarray [amp_bins + 1]
        - global_loudness_db: float
        - dynamic_range_db: float
    """
    song_path = Path(songPath)

    # Load audio as mono, preserve native sample rate
    # librosa returns float32 in [-1, 1]
    y, sr = librosa.load(song_path, sr=None, mono=True)

    # STFT parameters (keep in sync with your rendering expectations)
    n_fft = 2048
    hop_length = 1024

    # Compute STFT magnitude
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window="hann", center=True)
    magnitude = np.abs(stft).astype(np.float32)  # (freq_bins, frames)

    # Harmonic–percussive source separation for hue information
    H, P = librosa.decompose.hpss(stft)
    harmonic_mag = np.abs(H).astype(np.float32)
    percussive_mag = np.abs(P).astype(np.float32)
    hp_eps = 1e-10
    hp_ratio = harmonic_mag / (harmonic_mag + percussive_mag + hp_eps)

    # Magnitude in dB (for visualization / loudness-like behavior)
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max).astype(np.float32)

    # Compute A-weighted loudness per frame
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    freqs_safe = np.maximum(freqs, 1.0)  # or some small positive value
    A_weight_db = librosa.A_weighting(freqs_safe)
    A_weight_gain = 10.0 ** (A_weight_db / 20.0)
    magnitude_A = magnitude * A_weight_gain[:, np.newaxis]
    eps = 1e-10
    aw_rms = np.sqrt(np.mean(magnitude_A ** 2, axis=0) + eps).astype(np.float32)
    loudness_A_db = (20.0 * np.log10(aw_rms + eps)).astype(np.float32)
    global_loudness_A_db = float(loudness_A_db.mean())
    p95_A = float(np.percentile(loudness_A_db, 95))
    p5_A = float(np.percentile(loudness_A_db, 5))
    dynamic_range_A_db = p95_A - p5_A

    # Frame-wise RMS
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length, center=True)[0].astype(np.float32)

    # Loudness in dB from RMS (simple approximation)
    loudness_db = (20.0 * np.log10(rms + eps)).astype(np.float32)

    # Global loudness metric (mean loudness over frames)
    global_loudness_db = float(loudness_db.mean())

    # Dynamic range (95th - 5th percentile of loudness)
    p95 = float(np.percentile(loudness_db, 95))
    p5 = float(np.percentile(loudness_db, 5))
    dynamic_range_db = p95 - p5

    # Amplitude histogram of raw waveform (for trend mode)
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

    The point cloud encodes:
        - x: normalized time in [-1, 1]
        - y: log amplitude (dB-like) normalized to [-1, 1]
        - z: normalized frequency index in [-1, 1]
        - r,g,b,a: color derived solely from harmonic–percussive ratio.

    The output is a raw float32 binary file with layout:
        [x, y, z, r, g, b, a] per vertex.

    Note: Completely silent time frames (no energy across all frequencies) are omitted from the point cloud.
    """
    required_keys = ("magnitude", "hp_ratio")
    if not all(k in features for k in required_keys):
        print(f"[exportSongPointCloud] Missing required feature keys in song features; expected {required_keys}. Skipping point cloud export.")
        return

    magnitude = features["magnitude"]        # [freq_bins, frames]
    hp_ratio = features["hp_ratio"]          # [freq_bins, frames]

    freq_bins, frames = magnitude.shape

    # Use all frequency bins and time frames (may produce a large point cloud)
    if freq_bins <= 0 or frames <= 0:
        print("[exportSongPointCloud] Empty magnitude array; skipping.")
        return

    freqs_idx = np.arange(freq_bins, dtype=int)
    times_idx = np.arange(frames, dtype=int)

    # Amplitude per (freq, time)
    amp = magnitude[np.ix_(freqs_idx, times_idx)]  # [F, T]

    # Identify and drop completely silent time frames (no energy across all frequencies).
    col_max = amp.max(axis=0)  # [T]
    valid_mask = col_max > 0.0
    if not np.any(valid_mask):
        print("[exportSongPointCloud] All frames are silent; skipping.")
        return

    # Keep only non-silent time frames
    times_idx = times_idx[valid_mask]
    amp = amp[:, valid_mask]

    # Recompute frames count after filtering
    frames_valid = amp.shape[1]

    # Log amplitude (dB-like) for better dynamic range handling
    amp_eps = 1e-10
    amp_log = 20.0 * np.log10(amp + amp_eps)  # typically negative values

    # Normalize log amplitude globally over remaining points to [0,1]
    log_min = float(amp_log.min())
    log_max = float(amp_log.max())
    if log_max == log_min:
        amp_norm = np.zeros_like(amp_log, dtype=np.float32)
    else:
        amp_norm = ((amp_log - log_min) / (log_max - log_min)).astype(np.float32)

    # Map to [-1, 1] for NDC y-axis
    amp_ndc = amp_norm * 2.0 - 1.0

    # Normalized time (0..1) based on remaining (non-silent) frames, then to [-1,1]
    t_norm = times_idx / max(times_idx[-1], 1)
    t_ndc = t_norm * 2.0 - 1.0  # [T]

    # Normalized frequency index (0..1) then to [-1,1]
    f_norm = freqs_idx / max(freq_bins - 1, 1)
    f_ndc = f_norm * 2.0 - 1.0  # [F]

    # Broadcast positions: X = time, Y = log amplitude, Z = frequency
    X = np.broadcast_to(t_ndc, (freq_bins, frames_valid))
    Z = np.broadcast_to(f_ndc[:, None], (freq_bins, frames_valid))
    Y = amp_ndc

    # Sample harmonic–percussive ratio on the same (freq, time) grid
    hp = hp_ratio[np.ix_(freqs_idx, times_idx)]  # [F, frames_valid]

    # Color based solely on harmonic vs. percussive energy:
    # hp_ratio near 0 -> blue, near 1 -> red. Keep a small green accent.
    r = hp
    b = 1.0 - hp
    g = np.full_like(hp, 0.2)
    a = np.ones_like(r, dtype=np.float32)

    # Flatten into [N, 7] float32: x,y,z,r,g,b,a
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    Z_flat = Z.ravel()
    r_flat = r.ravel()
    g_flat = g.ravel()
    b_flat = b.ravel()
    a_flat = a.ravel()

    verts = np.stack(
        [X_flat, Y_flat, Z_flat, r_flat, g_flat, b_flat, a_flat],
        axis=1,
    ).astype(np.float32)

    out_path = Path(outPath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    verts.tofile(out_path)

    print(f"[exportSongPointCloud] Wrote point cloud: {out_path} (vertices: {verts.shape[0]})")


def extractSongFeatures() -> None:
    """Walk ./data/music, extract per-song features, and save them for later use.

    Directory layout (relative to the project root):
        ./data/music/<decade>/some_song.wav

    For each decade directory under ./data/music, this function:
        - Iterates over all .wav files in that decade
        - Extracts raw features needed for Song/Trend modes
        - Writes a compressed .npz file into:
              ./data/processed/<decade>_songs/<song_stem>_features.npz

    Saved fields per song include:
        - sample_rate: int
        - n_fft: int
        - hop_length: int
        - magnitude: 2D ndarray [freq_bins, frames]
        - hp_ratio: 2D ndarray [freq_bins, frames] (harmonic–percussive energy ratio per time–frequency bin)
        - rms: 1D ndarray [frames]
        - magnitude_db: 2D ndarray [freq_bins, frames]
        - loudness_db: 1D ndarray [frames]
        - loudness_A_db: 1D ndarray [frames] (A-weighted loudness per frame)
        - global_loudness_A_db: float (mean A-weighted loudness)
        - dynamic_range_A_db: float (A-weighted dynamic range)
        - amp_hist: 1D ndarray [amp_bins]
        - amp_bin_edges: 1D ndarray [amp_bins + 1]
        - global_loudness_db: float
        - dynamic_range_db: float
        - decade: str (directory name, e.g. "1980s")
        - song: str (original filename)
    """
    project_root = Path(__file__).resolve().parent
    music_root = project_root / "data" / "music"
    processed_root = project_root / "data" / "processed"

    if not music_root.exists():
        print(f"[extractSongFeatures] Music root not found: {music_root}")
        return

    # Ensure processed root exists
    processed_root.mkdir(parents=True, exist_ok=True)

    for decade_dir in sorted(music_root.iterdir()):
        if not decade_dir.is_dir():
            continue

        decade_label = decade_dir.name
        print(f"[extractSongFeatures] Processing decade: {decade_label}")

        decade_out_dir = processed_root / f"{decade_label}_songs"
        decade_out_dir.mkdir(parents=True, exist_ok=True)

        song_files = sorted(
            p for p in decade_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".wav", ".wave"}
        )

        if not song_files:
            print(f"[extractSongFeatures] No .wav files found in {decade_dir}")
            continue

        for song_path in song_files:
            print(f"[extractSongFeatures]   {song_path.name}")
            try:
                features = getSongFeatures(str(song_path))
            except Exception as exc:
                print(f"[extractSongFeatures]   ERROR processing {song_path.name}: {exc}")
                continue

            out_name = f"{song_path.stem}_features.npz"
            out_path = decade_out_dir / out_name
            bin_name = f"{song_path.stem}_points.bin"
            bin_path = decade_out_dir / bin_name

            # Add metadata and save
            np.savez_compressed(
                out_path,
                decade=np.array(decade_label),
                song=np.array(song_path.name),
                **features,
            )

            rel_out = out_path.relative_to(project_root)
            rel_bin_out = bin_path.relative_to(project_root)
            print(f"[extractSongFeatures]   -> {rel_out}")
            
            exportSongPointCloud(features, rel_bin_out)

        # After processing all songs for this decade, aggregate trend features.
        aggregateTrendFeatures(str(decade_out_dir))


def main() -> None:
    extractSongFeatures()


if __name__ == "__main__":
    main()
