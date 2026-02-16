"""
ICA 8-step workflow:
(1) Register free-breathing series with ANTsPy + draw lung mask (Napari) + bandpass filter
(2) ICA #1 to get dominant reference component
(3) Estimate per-voxel delays wrt reference
(4) Build delay map
(5) Realign voxel signals using delays
(6) ICA #2 on aligned data to isolate true physiological component
(7) Construct physiological signal from selected component
(8) Create ventilation/perfusion amplitude maps
"""

from __future__ import annotations
import os, tempfile

tmp = os.path.expanduser("~/ants_tmp")
os.makedirs(tmp, exist_ok=True)
os.environ["TMPDIR"] = tmp
os.environ["TMP"] = tmp
os.environ["TEMP"] = tmp
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

tempfile.tempdir = tmp
print("Using temp dir:", tempfile.gettempdir())

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy.signal import butter, filtfilt, correlate
from sklearn.decomposition import FastICA

try:
    import ants
    _HAVE_ANTS = True
except Exception:
    _HAVE_ANTS = False

try:
    import napari
    from skimage.segmentation import flood
    _HAVE_NAPARI = True
except Exception:
    _HAVE_NAPARI = False


# ----------------------------
# Registration + mask + filtering
# ----------------------------

def register_series(image_series: np.ndarray, ref_image: np.ndarray,
                                     params: Optional[dict] = None,) -> np.ndarray:
    """
    Registers each frame to ref_image using ANTsPy
    """
    if not _HAVE_ANTS:
        print(" ANTsPy isn't available -> using identity registration.")
        return image_series

    if params is None:
        params = {"type_of_transform": "SyN", "metric": ["CC"], "radius_or_number_of_bins": [4],
            "sampling_strategy": ["Regular"], "sampling_percentage": [0.25], "reg_iterations": (100, 100, 50),
            "smoothing_sigmas": (3, 2, 1), "shrink_factors": (4, 2, 1), "verbose": False}

    fixed_np_original = ref_image.astype(np.float32)
    fixed_ants_original = ants.from_numpy(fixed_np_original)

    min_val_fixed, max_val_fixed = float(fixed_np_original.min()), float(fixed_np_original.max())
    fixed_np_norm = (fixed_np_original - min_val_fixed) / (max_val_fixed - min_val_fixed + 1e-6)
    fixed_ants_norm = ants.from_numpy(fixed_np_norm.astype(np.float32))

    registered = []
    for i in range(image_series.shape[0]):
        moving_np_original = image_series[i].astype(np.float32)
        moving_ants_original = ants.from_numpy(moving_np_original)

        min_val_m, max_val_m = float(moving_np_original.min()), float(moving_np_original.max())
        moving_np_norm = (moving_np_original - min_val_m) / (max_val_m - min_val_m + 1e-6)
        moving_ants_norm = ants.from_numpy(moving_np_norm.astype(np.float32))

        reg = ants.registration(fixed=fixed_ants_norm, moving=moving_ants_norm, **params)

        warped_ants = ants.apply_transforms(fixed=fixed_ants_original, moving=moving_ants_original,
            transformlist=reg["fwdtransforms"])
        registered.append(warped_ants.numpy())

    return image_series


def make_two_ellipse_lung_mask(shape: Tuple[int, int]) -> np.ndarray:
    """
    Creates a simple "two-lung" binary mask using two ellipses.
    """
    Y, X = shape
    yy, xx = np.mgrid[0:Y, 0:X]

    # two ellipses
    cy = Y * 0.55
    cx1 = X * 0.38
    cx2 = X * 0.62
    ry = Y * 0.28
    rx = X * 0.18

    lung1 = ((yy - cy) / ry) ** 2 + ((xx - cx1) / rx) ** 2 <= 1.0
    lung2 = ((yy - cy) / ry) ** 2 + ((xx - cx2) / rx) ** 2 <= 1.0
    return lung1 | lung2


def draw_lung_mask_napari(ref_image: np.ndarray) -> np.ndarray:
    """
    Draw a lung ROI mask using Napari labels
        - Paint label 1 = lungs
        - Press 'g' to region-grow from newly painted pixels (flood fill) to speed up drawing
        - Close Napari to finalize
   !! If mask isn't available, it returns an automatic two-ellipse mask.
    """
    if not _HAVE_NAPARI:
        print("mask not available -> using automatic two-ellipse lung mask.")
        return make_two_ellipse_lung_mask(ref_image.shape)

    img = ref_image.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    viewer = napari.Viewer()
    viewer.add_image(img, name="Reference image")

    labels = np.zeros_like(img, dtype=np.uint8)
    label_layer = viewer.add_labels(labels, name="ROI labels (paint label 1 for lungs)")
    last_mask = np.zeros_like(labels, dtype=bool)

    @viewer.bind_key("g")
    def grow_roi(_viewer):
        nonlocal last_mask
        current_label = label_layer.selected_label
        if current_label != 1:
            print("Select label 1 (lungs) in the Napari labels UI first.")
            return

        current_mask = (label_layer.data == 1)
        new_paint = current_mask & (~last_mask)
        if not np.any(new_paint):
            print("No new painted region detected. Paint a small region first, then press 'g'.")
            return

        coords = np.argwhere(new_paint)
        seed = tuple(coords[len(coords) // 2])

        tolerance = 0.05
        region = flood(img, seed_point=seed, tolerance=tolerance, connectivity=1)

        new_labels = label_layer.data.copy()
        new_labels[region] = 1
        label_layer.data = new_labels

        last_mask = (label_layer.data == 1)
    napari.run()

    data = label_layer.data
    if not np.any(data == 1):
        print("[mask] No manual lung mask drawn -> using automatic two-ellipse mask.")
        return make_two_ellipse_lung_mask(ref_image.shape)

    return (data == 1)


def bandpass_filter_2d(data_2d: np.ndarray, fs: float, band: Tuple[float, float], order: int = 2) -> np.ndarray:
    lo, hi = band
    nyq = 0.5 * fs
    if not (0 < lo < hi < nyq):
        raise ValueError(
            f"Invalid band {band} for fs={fs} Hz (Nyquist={nyq} Hz). "
            f"Need 0 < lo < hi < {nyq}.")
    b, a = butter(order, [lo / nyq, hi / nyq], btype="bandpass")
    return filtfilt(b, a, data_2d, axis=0)


def extract_masked_matrix(series: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts voxels inside mask into a 2D matrix.
    """
    T, Y, X = series.shape
    flat_idx = np.flatnonzero(mask.reshape(-1))
    data_2d = series.reshape(T, -1)[:, flat_idx]
    return data_2d, flat_idx


# ----------------------------
#  ICA + selection + delay alignment
# ----------------------------

def run_ica(data_2d: np.ndarray, n_comp: int = 3, random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    ica = FastICA(n_components=n_comp, whiten="unit-variance", random_state=random_state, max_iter=1000)
    S = ica.fit_transform(data_2d)
    A = ica.mixing_
    return S, A


def select_components_by_band_peak(S: np.ndarray, fs: float, band: Tuple[float, float], kmax: int = 2,
    freq_tol_hz: float = 0.05, ratio_thresh: float = 0.75) -> tuple[list[int], list[tuple[int, float, float]]]:

    T, n_comp = S.shape
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)

    m = (freqs >= band[0]) & (freqs <= band[1])
    idx_band = np.where(m)[0]
    if idx_band.size == 0:
        raise ValueError(f"No FFT bins inside band={band} for fs={fs} and T={T}.")

    scores = []
    for k in range(n_comp):
        x = S[:, k] - S[:, k].mean()
        mag = np.abs(np.fft.rfft(x))

        mag_band = mag[idx_band]
        j = int(np.argmax(mag_band))
        peak_idx = int(idx_band[j])
        peak_freq = float(freqs[peak_idx])
        peak_mag = float(mag[peak_idx])

        scores.append((k, peak_freq, peak_mag))

    scores.sort(key=lambda z: z[2], reverse=True)

    k0, f0, p0 = scores[0]
    chosen = [k0]

    if kmax >= 2 and len(scores) > 1:
        k1, f1, p1 = scores[1]
        same_freq = abs(f1 - f0) <= freq_tol_hz
        strong_enough = p1 >= ratio_thresh * p0
        if same_freq and strong_enough:
            chosen.append(k1)

    return chosen, scores


def estimate_tau_from_phase_at_f0(data_2d: np.ndarray, fs: float, band_hz: tuple[float, float],
    ref: np.ndarray,) -> tuple[np.ndarray, float]:
    """
    Estimate per-voxel delay tau (seconds) relative to `ref`
    from phase difference at dominant frequency f0 
    """
    T, N = data_2d.shape
    t = np.arange(T, dtype=np.float32) / fs

    ref0 = (ref - ref.mean()) / (np.std(ref) + 1e-6)
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(ref0))

    m = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    idxs = np.where(m)[0]
    if idxs.size == 0:
        raise ValueError("Band has no FFT bins.")
    f0 = float(freqs[idxs[np.argmax(mag[idxs])]])
    w = 2.0 * np.pi * f0

    e = np.exp(-1j * w * t)

    z_ref = np.sum(ref0 * e)
    phi_ref = np.angle(z_ref)

    X = (data_2d - data_2d.mean(axis=0, keepdims=True)) / (np.std(data_2d, axis=0, keepdims=True) + 1e-6)

    z = np.sum(X * e[:, None], axis=0)
    dphi = np.angle(z) - phi_ref

    dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi

    tau = dphi / w  # seconds
    T0 = 1.0 / max(f0, 1e-6)
    tau = ((tau + 0.5 * T0) % T0) - 0.5 * T0

    return tau.astype(np.float32), f0


def apply_alignment_circular_fractional(data_2d: np.ndarray, tau_s: np.ndarray, fs: float) -> np.ndarray:
    T, N = data_2d.shape
    aligned = np.empty_like(data_2d, dtype=np.float32)
    for i in range(N):
        aligned[:, i] = circ_shift_frac_fft(data_2d[:, i], shift_samples=(-tau_s[i] * fs))
    return aligned


def fill_map(flat_idx: np.ndarray, values: np.ndarray, shape: Tuple[int, int],
             fill_value: float = np.nan, absval: bool = False) -> np.ndarray:
    Y, X = shape
    out = np.full(Y * X, fill_value, dtype=np.float32)
    v = values.astype(np.float32)
    if absval:
        v = np.abs(v)
    out[flat_idx] = v
    return out.reshape((Y, X))


def amplitude_map_from_mixing(flat_idx: np.ndarray, A: np.ndarray, comp_k: int, shape: Tuple[int, int]) -> np.ndarray:
    """ Converts ICA mixing weights for the selected component into a 2D amplitude map. """
    Y, X = shape
    out = np.zeros(Y * X, dtype=np.float32)
    out[flat_idx] = np.abs(A[:, comp_k]).astype(np.float32)
    return out.reshape((Y, X))


def combine_phase_split_components(S: np.ndarray, idx_a: int, idx_b: int, fs: float) -> np.ndarray:
    a = S[:, idx_a] - S[:, idx_a].mean()
    b = S[:, idx_b] - S[:, idx_b].mean()

    a /= (np.std(a) + 1e-6)
    b /= (np.std(b) + 1e-6)

    c = correlate(b, a, mode="full")
    lag = int(np.argmax(np.abs(c)) - (len(a) - 1))
    b_aligned = np.roll(b, -lag)

    if np.mean(b_aligned * a) < 0:
        b_aligned = -b_aligned

    ref = 0.5 * (a + b_aligned)
    ref -= ref.mean()
    return ref


def align_signal_to_truth(x: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, int]:
    x0 = x - x.mean()
    t0 = truth - truth.mean()
    c = correlate(x0, t0, mode="full")
    lag = int(np.argmax(np.abs(c)) - (len(x0) - 1))
    x_aligned = np.roll(x, -lag)
    # optional: fix sign
    if np.corrcoef(x_aligned, truth)[0, 1] < 0:
        x_aligned = -x_aligned
    return x_aligned, lag


def circ_shift_frac_fft(x: np.ndarray, shift_samples: float) -> np.ndarray:
    """
    Circular fractional shift of a 1D signal by shift_samples.
    Positive shift_samples delays the signal (moves it right).
    Negative shift_samples advances the signal.
    """
    x = np.asarray(x, dtype=np.float32)
    T = x.size
    X = np.fft.rfft(x)
    k = np.fft.rfftfreq(T)
    phase = np.exp(-2j * np.pi * k * shift_samples)
    y = np.fft.irfft(X * phase, n=T)
    return y.astype(np.float32)


def mean_abs_corr_to_ref(data_2d: np.ndarray, ref: np.ndarray) -> float:
    ref0 = (ref - ref.mean()) / (np.std(ref) + 1e-6)
    X = (data_2d - data_2d.mean(axis=0, keepdims=True)) / (np.std(data_2d, axis=0, keepdims=True) + 1e-6)
    c = np.mean(X * ref0[:, None], axis=0)
    return float(np.mean(np.abs(c)))


# ----------------------------
# Plotting
# ----------------------------

def plot_components(S, A, comp_indices, flat_idx, shape, fs, band, title, cmap="hot", theme: str|None=None,
                    raw_global = None, same_scale=True, sort_by="recon_amp", show_only_positive_freq=True):
    """
    Displays ICA components in a diagnostic plot:
            col 1: spatial map (mixing weights in image space)
            col 2: spectrum magnitude (FFT)
            col 3: timecourse
    """
    theme = (theme or "").lower()
    if theme in ("vent", "ventilation"):
        line_color = "tab:blue"
        band_color = "tab:blue"
    elif theme in ("perf", "perfusion"):
        line_color = "tab:red"
        band_color = "tab:red"
    else:
        line_color = None
        band_color = None

    T = S.shape[0]
    t = np.arange(T) / fs
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)

    recon = []
    max_fft_mag = 0.0

    for k in comp_indices:
        sig = S[:, k]

        basis = np.vstack([sig, np.ones_like(sig)]).T
        amp, dc = np.linalg.lstsq(basis, raw_global, rcond=None)[0]
        rec = sig * amp + dc

        mag = np.abs(np.fft.rfft(rec - np.mean(rec)))
        if show_only_positive_freq:
            pass

        max_fft_mag = max(max_fft_mag, float(np.max(mag)))

        recon.append({"k": k, "reconstruction": rec, "amp": float(amp),
            "dc": float(dc), "fft_mag": mag})

    if sort_by == "recon_amp":
        recon.sort(key=lambda d: float(np.max(np.abs(d["reconstruction"]))), reverse=True)
    elif sort_by == "band_fft_peak" and band is not None:
        f0, f1 = band
        mask = (freqs >= f0) & (freqs <= f1)
        recon.sort(key=lambda d: float(np.max(d["fft_mag"][mask])) if np.any(mask) else float(np.max(d["fft_mag"])),
            reverse=True)
    else:
        recon.sort(key=lambda d: float(np.max(d["fft_mag"])), reverse=True)

    if same_scale:
        fft_ylim = (0.0, max_fft_mag * 1.05 if max_fft_mag > 0 else 1.0)

        global_time_max = float(np.max(np.abs(raw_global)))
        for d in recon:
            global_time_max = max(global_time_max, float(np.max(np.abs(d["reconstruction"]))))

        time_ylim = (-global_time_max * 1.05, global_time_max * 1.05) if global_time_max > 0 else (-1.0, 1.0)
    else:
        fft_ylim = None
        time_ylim = None

    # ---- Plot ----
    n = len(recon)
    fig, axs = plt.subplots(n, 3, figsize=(10, 2.2 * n), squeeze=False)
    fig.suptitle(title, fontsize=14)

    for row, d in enumerate(recon):
        k = d["k"]
        rec = d["reconstruction"]
        mag = d["fft_mag"]

        # Col 1: map
        amp_map = amplitude_map_from_mixing(flat_idx, A, k, shape)
        im = axs[row, 0].imshow(amp_map, cmap=cmap)
        axs[row, 0].set_title(f"Comp. {k + 1} map")
        axs[row, 0].axis("off")
        fig.colorbar(im, ax=axs[row, 0], fraction=0.046, pad=0.04)

        # Col 2: spectrum of reconstructed signal
        if line_color is None:
            axs[row, 1].plot(freqs, mag)
            if band is not None:
                axs[row, 1].axvspan(band[0], band[1], alpha=0.2)
        else:
            axs[row, 1].plot(freqs, mag, color=line_color, linewidth=1.5)
            if band is not None:
                axs[row, 1].axvspan(band[0], band[1], alpha=0.15, color=band_color)

        axs[row, 1].set_ylabel("FFT Magn.")
        axs[row, 1].set_xlabel("Frequency [Hz]")
        if fft_ylim is not None:
            axs[row, 1].set_ylim(*fft_ylim)

        # Col 3: reconstruction fit vs raw_global (same scale)
        axs[row, 2].plot(t, raw_global, color="0.6", alpha=0.6, linewidth=1.0, label="Raw global")
        if line_color is None:
            axs[row, 2].plot(t, rec, linewidth=1.5, label=f"Comp {k + 1} fit")
        else:
            axs[row, 2].plot(t, rec, color=line_color, linewidth=1.5, label=f"Comp {k + 1} fit")

        axs[row, 2].set_ylabel("Signal")
        axs[row, 2].set_xlabel("Time [s]")
        if time_ylim is not None:
            axs[row, 2].set_ylim(*time_ylim)

    plt.tight_layout()
    plt.show()


def plot_physiological_outputs(t: np.ndarray, vent_phys: np.ndarray, perf_phys: np.ndarray, truth: dict | None = None):
    """
    Plots the extracted physiological signals (ventilation + perfusion).
    If truth is provided, overlays the ground truth waveforms.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axs[0].plot(t, vent_phys, label="Raw vent signal", color="gray")
    if truth is not None:
        axs[0].plot(t, truth["vent_sig"], "--", label="True vent (synthetic)", color="blue")
    axs[0].set_ylabel("Intensity [a.u.]")
    axs[0].set_title("Ventilation signal")
    axs[0].legend()

    axs[1].plot(t, perf_phys, label="Raw perf signal", color="gray")
    if truth is not None:
        axs[1].plot(t, truth["perf_sig"], "--", label="True perf (synthetic)", color="red")
    axs[1].set_ylabel("Intensity [a.u.]")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_title("Perfusion signal")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


# ----------------------------
# Workflow outputs
# ----------------------------
@dataclass
class WorkflowOutput:
    lung_mask: np.ndarray
    vent_delay_map_s: np.ndarray
    perf_delay_map_s: np.ndarray
    vent_amp_map: np.ndarray
    perf_amp_map: np.ndarray
    vent_selected_component: int
    perf_selected_component: int
    vent_phys: np.ndarray
    perf_phys: np.ndarray


# ----------------------------
# ICA 8-step workflow
# ----------------------------
def run_8_step_workflow(series: np.ndarray, fs: float, vent_band_hz: Tuple[float, float] = (0.05, 0.5),
    perf_band_hz: Tuple[float, float] = (0.8, 2.0), max_delay_s: float = 2.0, interactive_mask: bool = False,
    random_state: int = 0, ) -> WorkflowOutput:

    # -------------------------
    # (1) Registration + mask + filtering
    # -------------------------
    ref = series[0]
    reg_series = series

    if interactive_mask:
        lung_mask = draw_lung_mask_napari(reg_series.mean(axis=0))
    else:
        lung_mask = make_two_ellipse_lung_mask(reg_series.shape[1:])

    data_2d, flat_idx = extract_masked_matrix(reg_series, lung_mask)

    vent_filt = bandpass_filter_2d(data_2d, fs, vent_band_hz)
    perf_filt = bandpass_filter_2d(data_2d, fs, perf_band_hz)

    # -------------------------
    # (2) First ICA: dominant reference
    # -------------------------
    S_v1, A_v1 = run_ica(vent_filt, n_comp=3, random_state=random_state)
    vent_chosen, vent_scores = select_components_by_band_peak(S_v1, fs, vent_band_hz,
        kmax=2, freq_tol_hz=0.05, ratio_thresh=0.75)

    print("Vent ICA#1 chosen components:", vent_chosen)

    if len(vent_chosen) == 2:
        a, b = vent_chosen
        c = correlate(S_v1[:, b] - S_v1[:, b].mean(),
                      S_v1[:, a] - S_v1[:, a].mean(), mode="full")
        lag = int(np.argmax(np.abs(c)) - (len(S_v1) - 1))
        print("Vent ICA#1 phase-shift lag (s):", lag / fs)

    k_v_ref_a = vent_chosen[0]
    k_v_ref_b = vent_chosen[1] if len(vent_chosen) == 2 else None

    S_q1, A_q1 = run_ica(perf_filt, n_comp=3, random_state=random_state)

    perf_chosen, perf_scores = select_components_by_band_peak(S_q1, fs, perf_band_hz, kmax=2,
        freq_tol_hz=0.15, ratio_thresh=0.60)

    print("Perf ICA#1 chosen components:", perf_chosen)

    k_q_ref_a = perf_chosen[0]
    k_q_ref_b = perf_chosen[1] if len(perf_chosen) == 2 else None

    # Show diagnostic plots
    plot_components(S_v1, A_v1, (0, 1, 2), flat_idx, lung_mask.shape, fs, vent_band_hz,
                    "ICA Ventilation", cmap="hot", theme="vent", raw_global=vent_filt.mean(axis=1))

    plot_components(S_q1, A_q1, (0, 1, 2), flat_idx, lung_mask.shape, fs, perf_band_hz,
                    "ICA Perfusion", cmap="hot", theme="perf", raw_global=perf_filt.mean(axis=1))

    # -------------------------
    # (3) Delays wrt reference
    # -------------------------
    if k_v_ref_b is None:
        ref_v = S_v1[:, k_v_ref_a]
    else:
        ref_v = combine_phase_split_components(S_v1, k_v_ref_a, k_v_ref_b, fs)

    ref_v0 = (ref_v - ref_v.mean()) / (np.std(ref_v) + 1e-6)

    if np.corrcoef(vent_filt.mean(axis=1), ref_v0)[0, 1] < 0:
        ref_v0 = -ref_v0


    if k_q_ref_b is None:
        ref_q = S_q1[:, k_q_ref_a]
    else:
        ref_q = combine_phase_split_components(S_q1, k_q_ref_a, k_q_ref_b, fs)

    ref_q0 = (ref_q - ref_q.mean()) / (np.std(ref_q) + 1e-6)

    if np.corrcoef(perf_filt.mean(axis=1), ref_q0)[0, 1] < 0:
        ref_q0 = -ref_q0

    # Ventilation delay estimation
    vent_tau, f0v = estimate_tau_from_phase_at_f0(vent_filt, fs, vent_band_hz, ref_v0)
    T0v = 1.0 / f0v
    print(f"Vent f0={f0v:.3f} Hz, T0={T0v:.3f} s")

    # Perfusion delay estimation
    perf_tau, f0p = estimate_tau_from_phase_at_f0(perf_filt, fs, perf_band_hz, ref_q0)
    T0p = 1.0 / f0p
    print(f"Perf f0={f0p:.3f} Hz, T0={T0p:.3f} s")

    # -------------------------
    # (4) Delay maps
    # -------------------------
    vent_delay_map = fill_map(flat_idx, vent_tau, lung_mask.shape, fill_value=np.nan)
    perf_delay_map = fill_map(flat_idx, perf_tau, lung_mask.shape, fill_value=np.nan)

    # -------------------------
    # (5) Realignment
    # -------------------------
    # --- Vent alignment:
    vent_aligned_m = apply_alignment_circular_fractional(vent_filt, vent_tau, fs)
    vent_aligned_p = apply_alignment_circular_fractional(vent_filt, -vent_tau, fs)

    cm = mean_abs_corr_to_ref(vent_aligned_m, ref_v0)
    cp = mean_abs_corr_to_ref(vent_aligned_p, ref_v0)

    if cp > cm:
        vent_tau = -vent_tau
        vent_aligned = vent_aligned_p
    else:
        vent_aligned = vent_aligned_m

    print("Vent mean |corr| pre:", mean_abs_corr_to_ref(vent_filt, ref_v0))
    print("Vent mean |corr| post:", mean_abs_corr_to_ref(vent_aligned, ref_v0))

    ref0 = (ref_v0 - ref_v0.mean()) / (np.std(ref_v0) + 1e-6)
    X = (vent_aligned - vent_aligned.mean(axis=0, keepdims=True)) / (np.std(vent_aligned, axis=0, keepdims=True) + 1e-6)
    c = np.mean(X * ref0[:, None], axis=0)
    vent_aligned[:, c < 0] *= -1.0

    # --- Perf alignment:
    perf_aligned_m = apply_alignment_circular_fractional(perf_filt, perf_tau, fs)
    perf_aligned_p = apply_alignment_circular_fractional(perf_filt, -perf_tau, fs)

    cm = mean_abs_corr_to_ref(perf_aligned_m, ref_q0)
    cp = mean_abs_corr_to_ref(perf_aligned_p, ref_q0)

    if cp > cm:
        perf_tau = -perf_tau
        perf_aligned = perf_aligned_p
    else:
        perf_aligned = perf_aligned_m

    print("Perf mean |corr| pre:", mean_abs_corr_to_ref(perf_filt, ref_q0))
    print("Perf mean |corr| post:", mean_abs_corr_to_ref(perf_aligned, ref_q0))

    ref0p = (ref_q0 - ref_q0.mean()) / (np.std(ref_q0) + 1e-6)
    Xp = (perf_aligned - perf_aligned.mean(axis=0, keepdims=True)) / (
                np.std(perf_aligned, axis=0, keepdims=True) + 1e-6)
    cpix = np.mean(Xp * ref0p[:, None], axis=0)
    perf_aligned[:, cpix < 0] *= -1.0

    # -------------------------
    # (6) Second ICA on aligned data
    # -------------------------
    S_v2, A_v2 = run_ica(vent_aligned, n_comp=3, random_state=random_state)
    vent2_chosen, vent2_scores = select_components_by_band_peak(S_v2, fs, vent_band_hz, kmax=1, freq_tol_hz=0.05, ratio_thresh=0.75)

    print("Vent ICA#2 chosen components:", vent2_chosen)
    k_v_true = vent2_chosen[0]

    S_q2, A_q2 = run_ica(perf_aligned, n_comp=3, random_state=random_state)
    perf2_chosen, perf2_scores = select_components_by_band_peak(S_q2, fs, perf_band_hz, kmax=1, freq_tol_hz=0.15, ratio_thresh=0.75)

    print("Perf ICA#2 chosen components:", perf2_chosen)
    k_q_true = perf2_chosen[0]

    # (ICA #2)
    plot_components(S_v2, A_v2, (0, 1, 2), flat_idx, lung_mask.shape, fs, vent_band_hz, "ICA Ventilation (aligned)",
                    cmap="hot", theme="vent", raw_global=vent_aligned.mean(axis=1))
    plot_components(S_q2, A_q2, (0, 1, 2), flat_idx, lung_mask.shape, fs, perf_band_hz, "ICA Perfusion (aligned)",
                    cmap="hot", theme="perf", raw_global=perf_aligned.mean(axis=1))

    # -------------------------
    # (7) Physiological signal
    # -------------------------
    vent_phys = S_v2[:, k_v_true]
    perf_phys = S_q2[:, k_q_true]

    # -------------------------
    # (8) Maps from amplitude of constructed aligned signals
    # -------------------------
    vent_amp_map = fill_map(flat_idx, A_v2[:, k_v_true], lung_mask.shape, fill_value=0.0, absval=True)
    perf_amp_map = fill_map(flat_idx, A_q2[:, k_q_true], lung_mask.shape, fill_value=0.0, absval=True)

    return WorkflowOutput(lung_mask=lung_mask, vent_delay_map_s=vent_delay_map, perf_delay_map_s=perf_delay_map,
        vent_amp_map=vent_amp_map, perf_amp_map=perf_amp_map, vent_selected_component=k_v_true,
        perf_selected_component=k_q_true, vent_phys=vent_phys, perf_phys=perf_phys)


# ----------------------------
# Synthetic data
# ----------------------------
def make_synthetic_mri_series(T: int = 500, Y: int = 96, X: int = 96, fs: float = 1.0, seed: int = 0) \
        -> tuple[np.ndarray, dict]:
    """
        Creates a synthetic MRI-series with:
            - two-lung mask region
            - ventilation oscillation (low frequency)
            - perfusion oscillation (higher frequency)
            - random noise background
    """
    rng = np.random.default_rng(seed)
    series = rng.normal(0, 1, size=(T, Y, X)).astype(np.float32) * 0.15

    lung_mask = make_two_ellipse_lung_mask((Y, X))
    yy, xx = np.mgrid[0:Y, 0:X]

    t = np.arange(T) / fs
    vent_sig = np.sin(2 * np.pi * 0.20 * t)  # 0.20 Hz
    perf_sig = np.sin(2 * np.pi * 1.20 * t)  # 1.20 Hz
    nuisance = 0.5 * np.sin(2 * np.pi * 0.02 * t)

    vent_w = (yy / (Y - 1)).astype(np.float32)
    perf_w = np.exp(-(((yy - Y * 0.55) / (Y * 0.25)) ** 2 + ((xx - X * 0.5) / (X * 0.30)) ** 2)).astype(np.float32)
    noise_w = rng.normal(0, 1, size=(Y, X)).astype(np.float32)
    noise_w = (noise_w - noise_w.min()) / (noise_w.max() - noise_w.min() + 1e-6)

    vent_w = vent_w * lung_mask
    perf_w = perf_w * lung_mask
    noise_w = noise_w * lung_mask

    if vent_w.max() > 0: vent_w /= vent_w.max()
    if perf_w.max() > 0: perf_w /= perf_w.max()
    if noise_w.max() > 0: noise_w /= noise_w.max()

    early = lung_mask & (yy < Y * 0.55)  # upper lung = earlier arrival
    late = lung_mask & (yy >= Y * 0.55)  # lower lung = later arrival

    vent_delay_s = np.zeros((Y, X), dtype=np.float32)
    perf_delay_s = np.zeros((Y, X), dtype=np.float32)

    vent_delay_s[early] = -0.6
    vent_delay_s[late] = +0.6

    perf_delay_s[early] = -0.4
    perf_delay_s[late] = +0.4

    vent_shift = np.round(vent_delay_s * fs).astype(int)
    perf_shift = np.round(perf_delay_s * fs).astype(int)

    for y in range(Y):
        for x in range(X):
            if not lung_mask[y, x]:
                continue
            vs = np.roll(vent_sig, vent_shift[y, x])
            ps = np.roll(perf_sig, perf_shift[y, x])
            ns = nuisance

            series[:, y, x] += (0.9 * vent_w[y, x] * vs + 0.35 * perf_w[y, x] * ps +
                    0.3 * noise_w[y, x] * ns).astype(np.float32)

    grad = (xx / max(X - 1, 1)).astype(np.float32) * 0.05
    series += grad[None, :, :]

    truth = {"t": t, "vent_sig": vent_sig, "perf_sig": perf_sig, "nuisance": nuisance}

    return series, truth




if __name__ == "__main__":
    fs = 5.0
    series, truth = make_synthetic_mri_series(T=500, Y=96, X=96, fs=fs, seed=0)

    out = run_8_step_workflow(series=series, fs=fs, vent_band_hz=(0.05, 0.5),
        perf_band_hz=(0.8, 2.0),  max_delay_s=2.0, interactive_mask=False,  # set True to draw lungs in Napari
        random_state=0)

    t = truth["t"]
    out_vent_plot, lag_v = align_signal_to_truth(out.vent_phys, truth["vent_sig"])
    out_perf_plot, lag_p = align_signal_to_truth(out.perf_phys, truth["perf_sig"])

    plot_physiological_outputs(truth["t"], out_vent_plot, out_perf_plot, truth=truth)

    print("Selected ventilation component (ICA #2):", out.vent_selected_component)
    print("Selected perfusion component (ICA #2):", out.perf_selected_component)
