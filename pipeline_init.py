# %% 
# General imports
import os
from itertools import combinations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm  # same flavour as SpikeInterface's chunk progress bars

# SI packages
import spikeinterface.full as si
from probeinterface.plotting import plot_probe
import spikeinterface.preprocessing as spre
import spikeinterface.widgets as sw
import spikeinterface.curation as sc

# Default for plotting
params = {  'legend.fontsize': 'large',
            'legend.title_fontsize': 'large',
            'axes.labelsize': 'large',
            'axes.titlesize':'x-large',
            'xtick.labelsize':'large',
            'ytick.labelsize':'large',
            'axes.linewidth': 2,
            'lines.linewidth': 4,
            'xtick.major.width': 2,
            'ytick.major.width': 2,
            }
plt.rcParams.update(params)

# CPU parallelism for SpikeInterface's chunked processing (saving, analyzer extensions...).
# These steps are numpy-only, so this does not affect the GPU: Kilosort4 runs on the GPU
# through torch_device in run_sorter below.
n_cpus = os.cpu_count()
n_jobs = max(1, n_cpus - 4)
# Threads instead of processes: on Windows the process pool has to "spawn" fresh Python
# workers, which is slower to start and can hang silently inside Jupyter / the VS Code
# interactive window. The heavy numpy/scipy work releases the GIL, so threads are as fast or faster here.
job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=True, pool_engine="thread")
si.set_global_job_kwargs(**job_kwargs)  # also used by steps that don't get job_kwargs explicitly

# %%

"""
.##........#######.....###....########..####.##....##..######..
.##.......##.....##...##.##...##.....##..##..###...##.##....##.
.##.......##.....##..##...##..##.....##..##..####..##.##.......
.##.......##.....##.##.....##.##.....##..##..##.##.##.##...####
.##.......##.....##.#########.##.....##..##..##..####.##....##.
.##.......##.....##.##.....##.##.....##..##..##...###.##....##.
.########..#######..##.....##.########..####.##....##..######..
"""

# This should be the path to the folder containing
# the data (.bin and .meta files linked above). 
path_to_data_folder = Path("D:/tests_run/G_first_try/sofias_cable_animal_I_g0/sofias_cable_animal_I_g0_imec1")

def read_spikeglx_ap(folder):
    # NP2.0 quad-base probes (NP2020) save 1536 AP + 4 sync channels (snsApLfSy=1536,0,4).
    # neo only strips the sync channel when there is exactly one, so get_traces() without
    # channel_ids returns all 1540 columns while the recording reports 1536 channels
    # (-> broadcast errors e.g. in detect_bad_channels). Selecting the channels explicitly
    # makes every get_traces() call request the 1536 AP channels only.
    # The probe (imec0, imec1, ...) comes from the .ap.meta file name, so any probe folder works.
    (meta_file,) = Path(folder).glob("*.ap.meta")
    rec = si.read_spikeglx(folder, stream_id=meta_file.name.split(".")[-3] + ".ap")
    return rec.select_channels(rec.channel_ids)

def detect_bad_channels_per_shank(rec, **kwargs):
    # coherence+psd compares each channel with its neighbours along depth. On a multi-shank
    # probe, ordering all channels by (x, y) interleaves the shanks, so run it per shank
    # (a single-shank probe has one group, so this is a single call).
    # Labels are returned in the channel order of `rec`.
    bad_channel_ids = []
    channel_labels = np.empty(rec.get_num_channels(), dtype=object)
    for shank_rec in tqdm(rec.split_by("group").values(), desc="Bad channel detection (shanks)"):
        shank_bad_ids, shank_labels = spre.detect_bad_channels(shank_rec, **kwargs)
        bad_channel_ids.extend(shank_bad_ids)
        channel_labels[rec.ids_to_indices(shank_rec.channel_ids)] = shank_labels
    return np.array(bad_channel_ids), channel_labels

recording = read_spikeglx_ap(path_to_data_folder)

print(recording)

print(
    [method for method in dir(recording) if method.startswith("get")]
)

# Plotting probe and see geometry
probe = recording.get_probe()

plot_probe(probe)
plt.show()

# Shanks are read from the "group" channel property (one group per shank; a single-shank
# probe has a single group 0). Everything below is split by group, so it works for both.
n_shanks = len(np.unique(recording.get_property("group")))
print(f"{n_shanks} shank(s) detected:", "multi-shank" if n_shanks > 1 else "single-shank", "probe")

# 0.1 seconds of data from our raw recording
start_time = recording.get_times()[0]

si.plot_traces(
    recording,
    return_in_uV=True, # to ensure the data is returned in microvolts
    order_channel_by_depth=True
)
plt.show()

# Checking for bad channels 
bad_channel_ids, channel_labels = detect_bad_channels_per_shank(recording, method='coherence+psd')

print('bad_channel_ids', bad_channel_ids)
print('channel_labels', channel_labels)
print('\n Percentage of bad channels:', len(bad_channel_ids) / recording.get_num_channels() * 100)

good_chn_recording = recording.remove_channels(bad_channel_ids)

# Cutting if needed for faster testing (e.g. 5 minutes)
# fs = recording.get_sampling_frequency()
# recording_sub = recording.frame_slice(start_frame=0*fs, end_frame=300*fs)
# recording_sub

# %%
"""
.########..########..########.........########..########...#######...######.
.##.....##.##.....##.##...............##.....##.##.....##.##.....##.##....##
.##.....##.##.....##.##...............##.....##.##.....##.##.....##.##......
.########..########..######...#######.########..########..##.....##.##......
.##........##...##...##...............##........##...##...##.....##.##......
.##........##....##..##...............##........##....##..##.....##.##....##
.##........##.....##.########.........##........##.....##..#######...######.
"""

#  Pre-processing IBL method (phase_shift, bandpass_filter, common_reference)
phase_shift_recording = si.phase_shift(good_chn_recording)

filtered_recording = si.bandpass_filter(
    phase_shift_recording, freq_min=300, freq_max=6000
)

cmr_recording = si.common_reference(
    filtered_recording, operator="median"
)

print(phase_shift_recording)
print(filtered_recording)
print(cmr_recording)

# Actual pre-processed plots
si.plot_traces(
    cmr_recording,
    return_in_uV=True,
    order_channel_by_depth=True
)
plt.show()

# Splitting and plotting data by shank (a single-shank probe gives one entry, {0: recording})
split_recording = cmr_recording.split_by("group")
print(split_recording) # check if the groups match the ones defined in your geometry!

# squeeze=False so axes is always 2D, also with a single shank
fig, axes = plt.subplots(1, len(split_recording), squeeze=False)

for ax, (shank_idx, shank_recording) in zip(axes[0], split_recording.items()):

    si.plot_traces(
        shank_recording,
        return_in_uV=True,
        order_channel_by_depth=True,
        ax=ax
    )
    ax.set_title(f"Shank: {shank_idx}")

# Save preprocessed recording here
# One saved recording per shank, keyed by shank index (same keys as split_recording).
# A single-shank probe is saved the same way, as preprocessed_/shank_0.
saved_recordings = {}
for shank_idx, shank_recording in tqdm(split_recording.items(), desc="Saving preprocessed (shanks)"):

    saved_recordings[shank_idx] = shank_recording.save(folder=path_to_data_folder / "preprocessed_" / f"shank_{shank_idx}", **job_kwargs, overwrite=True)

# Motion correction - DOESNT APPLY CLEANUP, JUST FOR VIZ
# Run on the saved shanks rather than split_recording: reading through the lazy chain would
# recompute phase_shift + bandpass + CMR for every chunk (~85% of the motion correction time).
saved_corrected, motion_info = {}, {}
for shank_idx, shank_recording in tqdm(saved_recordings.items(), desc="Motion correction (shanks)"):

    saved_corrected[shank_idx], motion_info[shank_idx] = spre.correct_motion(
    shank_recording,
    preset="dredge_fast",
    output_motion_info=True,
    # turn off non-rigid estimation
    estimate_motion_kwargs={"rigid": True},
    **job_kwargs,)  # parallel peak detection/localization, with chunk progress bars

    tqdm.write(str(saved_corrected[shank_idx]))  # print without breaking the progress bars


# Display the motion output: drift map + estimated motion over depth/time, one figure per shank
for shank_idx in motion_info:
    fig = plt.figure(figsize=(14, 8))
    sw.plot_motion_info(
        motion_info[shank_idx],
        recording=saved_corrected[shank_idx],
        figure=fig,
        color_amplitude=True,
        amplitude_cmap="inferno",
        scatter_decimate=10,
    )
    fig.suptitle(f"Motion correction output - shank {shank_idx}")

# %% 

"""
..######...#######..########..########.####.##....##..######..
.##....##.##.....##.##.....##....##.....##..###...##.##....##.
.##.......##.....##.##.....##....##.....##..####..##.##.......
..######..##.....##.########.....##.....##..##.##.##.##...####
.......##.##.....##.##...##......##.....##..##..####.##....##.
.##....##.##.....##.##....##.....##.....##..##...###.##....##.
..######...#######..##.....##....##....####.##....##..######..
"""

# Sorting via Kilosort4
# Reload the preprocessed shanks saved above, so this cell can run in a fresh kernel
# (only the first cell is needed before it). Keys are the shank indices, as in split_recording.
saved_recordings = {
    int(shank_folder.name.removeprefix("shank_")): si.load(shank_folder)
    for shank_folder in sorted((path_to_data_folder / "preprocessed_").glob("shank_*"))
}
print(saved_recordings)

# One sorting per shank, keyed by shank index, run on the saved preprocessed data
# (so the preprocessing chain is not recomputed on the fly). A single-shank probe is a
# single run, saved in kilosort4_output_shank0.
# CHANGE IN CASE YOU WANT TO SORT ALL OF THE SHANKS TOGETHER
# (e.g. if you want to allow units to span shanks)
sortings = {}
for shank_idx, shank_recording in tqdm(saved_recordings.items(), desc="Kilosort4 (shanks)"):

    sortings[shank_idx] = si.run_sorter(
        "kilosort4",
        shank_recording,
        verbose=True,  # shows Kilosort4's own logs and progress bars (hidden otherwise)
        torch_device="cuda",  # fail loudly instead of silently falling back to the CPU
        clear_cache=False,  # True frees GPU memory between steps but is slower; set True only on CUDA OOM
        folder=path_to_data_folder / f"kilosort4_output_shank{shank_idx}",
        batch_size=30000, # For a 4GB GPU, default is 60k, lambda can hold more
        remove_existing_folder=True,
        do_CAR=False, # avoid double-referencing, since we already did it in the preprocessing step
        highpass_cutoff=0.1,
    )

for shank_idx, sorting in sortings.items():
    print(f"Shank {shank_idx}: {sorting}")

# %%

"""
....###....##....##....###....##.......##....##.########.########.########..
...##.##...###...##...##.##...##........##..##.......##..##.......##.....##.
..##...##..####..##..##...##..##.........####.......##...##.......##.....##.
.##.....##.##.##.##.##.....##.##..........##.......##....######...########..
.#########.##..####.#########.##..........##......##.....##.......##...##...
.##.....##.##...###.##.....##.##..........##.....##......##.......##....##..
.##.....##.##....##.##.....##.########....##....########.########.##.....##.
"""

# Analyzer for the Kilosort4 sorting: built once, saved to disk, reloaded afterwards
# Use it standalone when you don't want to re-run the sorting, it reads the Kilosort4
# outputs (if available) and the preprocessed shanks saved above.
analyzer_folder = path_to_data_folder / "analyzer_KS4"

if analyzer_folder.exists():
    analyzer = si.load_sorting_analyzer(analyzer_folder)
else:
    saved_recordings = {
        int(shank_folder.name.removeprefix("shank_")): si.load(shank_folder)
        for shank_folder in sorted((path_to_data_folder / "preprocessed_").glob("shank_*"))
    }
    # Kilosort4 can place a few spikes just past the end of the recording, which breaks
    # waveform extraction, so drop them
    sortings = {
        shank_idx: si.remove_excess_spikes(
            si.read_sorter_folder(path_to_data_folder / f"kilosort4_output_shank{shank_idx}"),
            shank_recording,
        )
        for shank_idx, shank_recording in saved_recordings.items()
    }
    # Multi-shank: one analyzer for all shanks (units get unique ids, shank in the
    # "aggregation_key" property). Default radius sparsity (~25 channels per unit) keeps each
    # unit on its own shank.
    # Single-shank: the shank's sorting and recording are passed directly, so unit ids stay
    # the Kilosort4 cluster ids and the recording stays a single binary file.
    # The binary_folder format keeps extensions on disk instead of RAM and saves them as they are computed.
    if len(saved_recordings) > 1:
        analyzer_sorting, analyzer_recording = sortings, saved_recordings
    else:
        (shank_idx,) = saved_recordings
        analyzer_sorting, analyzer_recording = sortings[shank_idx], saved_recordings[shank_idx]

    analyzer = si.create_sorting_analyzer(
        sorting=analyzer_sorting,
        recording=analyzer_recording,
        format="binary_folder",
        folder=analyzer_folder,
        **job_kwargs,
    )

print(analyzer)

# %%

"""
.##........#######.....###....########......######...#######..########..########
.##.......##.....##...##.##...##.....##....##....##.##.....##.##.....##....##...
.##.......##.....##..##...##..##.....##....##.......##.....##.##.....##....##...
.##.......##.....##.##.....##.##.....##.....######..##.....##.########.....##...
.##.......##.....##.#########.##.....##..........##.##.....##.##...##......##...
.##.......##.....##.##.....##.##.....##....##....##.##.....##.##....##.....##...
.########..#######..##.....##.########......######...#######..##.....##....##...
"""

# Alternative: load an existing sorting (Kilosort4 / IBL-sorter) instead of re-sorting
# Works on any Kilosort/Phy-style folder (spike_times.npy, spike_clusters.npy, params.py, ...),
# An ALF-converted IBL folder (spikes.samples.npy, clusters.*.npy) is also accepted.
path_to_sorting_output = Path("D:\\Prannay_Data\\Behav_2-8_11268_day7_sal_dls\\myRun22_g0\\myRun22_g0_imec0")  # <- folder with params.py
path_to_raw_data = path_to_sorting_output  # <- folder with the .ap.bin/.ap.meta it was sorted from
# The analyzer is built once and saved with the sorting, then reloaded with the QC extensions
# computed so far. It stores the raw .bin path relative to this folder, so the data folder can be moved.
loaded_analyzer_folder = path_to_sorting_output / "analyzer_SI"

def load_existing_sorting(sorter_folder, sampling_frequency):
    sorter_folder = Path(sorter_folder)
    if (sorter_folder / "params.py").exists():
        print(f"Loading Kilosort/Phy output from {sorter_folder}")
        return si.read_kilosort(sorter_folder, remove_empty_units=True)
    if (sorter_folder / "spikes.samples.npy").exists():
        print(f"Loading ALF output from {sorter_folder}")
        return si.read_alf_sorting(sorter_folder, sampling_frequency=sampling_frequency)
    # e.g. an IBL-sorter folder that only holds the _iblqc_* files
    candidates = sorted(p.parent for p in sorter_folder.rglob("params.py"))
    raise FileNotFoundError(
        f"No params.py or spikes.samples.npy in {sorter_folder}. "
        f"Sorting outputs found in subfolders: {candidates}"
    )

def load_main_channel_ids(sorter_folder, sorting, recording):
    # Main channel of each unit, read from the sorter output. Without it, create_sorting_analyzer
    # estimates it from ~100 spikes per unit, which reads through the whole .bin (hours from a USB drive).
    # `recording` is the raw recording the sorter ran on: the sorter's channel indices refer to it.
    # Returns None when the folder has neither templates nor cluster channels (the analyzer then estimates it).
    sorter_folder = Path(sorter_folder)
    if (sorter_folder / "templates.npy").exists():
        # Kilosort/Phy: unwhiten the templates and take the channel with the largest absolute peak
        # (create_sorting_analyzer's own definition: peak_sign="both", peak_mode="extremum")
        templates = np.load(sorter_folder / "templates.npy")  # (templates, samples, sorter channels)
        n_templates, n_samples, n_channels = templates.shape
        # one 2D matrix product (a 3D @ is ~100x slower here)
        templates = templates.reshape(-1, n_channels) @ np.load(sorter_folder / "whitening_mat_inv.npy").T
        templates = templates.reshape(n_templates, n_samples, n_channels)
        channel_map = np.load(sorter_folder / "channel_map.npy").ravel()
        template_main_channels = channel_map[np.abs(templates).max(axis=1).argmax(axis=1)]
        # Each unit takes the template most of its spikes come from: its own template for raw
        # Kilosort output, and still defined for clusters merged/split in Phy
        spike_clusters = np.load(sorter_folder / "spike_clusters.npy").ravel().astype(np.int64)
        spike_templates = np.load(sorter_folder / "spike_templates.npy").ravel()
        pairs, counts = np.unique(spike_clusters * n_templates + spike_templates, return_counts=True)
        clusters, cluster_templates = pairs // n_templates, pairs % n_templates
        order = np.lexsort((-counts, clusters))  # per cluster, most frequent template first
        first_clusters, first = np.unique(clusters[order], return_index=True)
        unit_templates = dict(zip(first_clusters, cluster_templates[order][first]))
        main_channel_indices = template_main_channels[[unit_templates[unit_id] for unit_id in sorting.unit_ids]]
    elif (sorter_folder / "clusters.channels.npy").exists():
        # ALF: unit ids index directly into the clusters.* arrays
        main_channel_indices = np.load(sorter_folder / "clusters.channels.npy")[sorting.unit_ids]
    else:
        return None
    return recording.channel_ids[main_channel_indices]

if loaded_analyzer_folder.exists():
    analyzer = si.load_sorting_analyzer(loaded_analyzer_folder)
else:
    loaded_recording = read_spikeglx_ap(path_to_raw_data)
    sorting = load_existing_sorting(path_to_sorting_output, loaded_recording.get_sampling_frequency())
    loaded_main_channel_ids = load_main_channel_ids(path_to_sorting_output, sorting, loaded_recording)
    print(loaded_recording)
    print(sorting)

    # Preprocess the full probe like Kilosort4 did (highpass + CAR), interpolating bad
    # channels instead of removing them so every unit keeps its peak channel
    loaded_preprocessed = si.highpass_filter(loaded_recording, freq_min=300.0)
    loaded_preprocessed = si.phase_shift(loaded_preprocessed)
    loaded_bad_channel_ids, _ = detect_bad_channels_per_shank(loaded_preprocessed, method="coherence+psd")
    print("bad_channel_ids", loaded_bad_channel_ids)
    loaded_preprocessed = spre.interpolate_bad_channels(loaded_preprocessed, loaded_bad_channel_ids)

    # Multi-shank: reference each shank separately, then re-join them into one recording.
    # Single-shank: reference the whole probe.
    loaded_shank_recordings = loaded_preprocessed.split_by("group")
    if len(loaded_shank_recordings) > 1:
        loaded_preprocessed = si.aggregate_channels(
            [si.common_reference(shank_rec, operator="median") for shank_rec in loaded_shank_recordings.values()]
        )
    else:
        loaded_preprocessed = si.common_reference(loaded_preprocessed, operator="median")

    # Sanity checks: same sampling rate and spikes inside the recording. Kilosort can place a few
    # spikes just past the end (dropped below); spikes far past it mean the .bin doesn't match the sorting.
    assert np.isclose(sorting.get_sampling_frequency(), loaded_preprocessed.get_sampling_frequency()), (
        f"Sampling frequencies differ: sorting {sorting.get_sampling_frequency()} Hz, "
        f"recording {loaded_preprocessed.get_sampling_frequency()} Hz"
    )
    last_spike_frame = sorting.to_spike_vector()["sample_index"].max()
    assert last_spike_frame < loaded_preprocessed.get_num_samples() + loaded_preprocessed.get_sampling_frequency(), (
        "Spikes fall more than 1 s after the end of the recording: is this the right .bin for this sorting?"
    )
    sorting = si.remove_excess_spikes(sorting, loaded_preprocessed)

    analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=loaded_preprocessed,
        sparse=True,
        format="binary_folder",  # extensions saved to disk as they are computed
        folder=loaded_analyzer_folder,
        # Main channels from the sorter output, so building the analyzer doesn't read the recording
        # (by channel id: aggregate_channels above can reorder the channels)
        main_channel_indices=(
            None if loaded_main_channel_ids is None else loaded_preprocessed.ids_to_indices(loaded_main_channel_ids)
        ),
        **job_kwargs
    )
print(analyzer)

# %% 

"""
..#######...######.
.##.....##.##....##
.##.....##.##......
.##.....##.##......
.##..##.##.##......
.##....##..##....##
..#####.##..######.
"""

# Quality control with Bombcell
extensions_dict = {
    'random_spikes': {},
    'templates': {},
    'noise_levels': {},
    'correlograms': {},
    'spike_amplitudes': {},
    'spike_locations': {},
    'unit_locations': {},
    'template_similarity': {},
    'isi_histograms': {},
    'template_metrics': {},
    'quality_metrics': {'skip_pc_metrics': True},  # PCA-based metrics are slow and not used by Bombcell
}

# Only compute what is missing, so a reloaded analyzer resumes where it stopped.
# Computing an extension deletes the ones that depend on it (e.g. spike_amplitudes ->
# quality_metrics), so loop until nothing is missing.
while missing_extensions := {name: params for name, params in extensions_dict.items() if not analyzer.has_extension(name)}:
    print("computing:", list(missing_extensions))
    analyzer.compute(missing_extensions, **job_kwargs)

# BOMBSHELL !!
bombcell_default_thresholds = sc.bombcell_get_default_thresholds()

bombcell_labels = sc.bombcell_label_units(analyzer,thresholds=bombcell_default_thresholds)

bombcell_labels["bombcell_label"].value_counts()

w = sw.plot_unit_labels(analyzer, bombcell_labels["bombcell_label"], ylims=(-300, 100))
w.figure.suptitle("Bombcell labeling")

# Check the distribution of the metrics used for labeling
_ = sw.plot_metric_histograms(analyzer, bombcell_default_thresholds, figsize=(15, 10))

# Check the UPSET plot of the labels
_ = sw.plot_bombcell_labels_upset(analyzer, unit_labels=bombcell_labels["bombcell_label"], 
            thresholds=bombcell_default_thresholds, unit_labels_to_plot=["noise", "mua"])


# %%
"""
.########.##.....##.########...#######..########..########
.##........##...##..##.....##.##.....##.##.....##....##...
.##.........##.##...##.....##.##.....##.##.....##....##...
.######......###....########..##.....##.########.....##...
.##.........##.##...##........##.....##.##...##......##...
.##........##...##..##........##.....##.##....##.....##...
.########.##.....##.##.........#######..##.....##....##...
"""

# One Phy folder for the whole probe, from the analyzer (all shanks; Phy gets the
# shank of each channel from channel_groups.npy). On a multi-shank probe the analyzer's
# recording is an aggregation of the per-shank saved recordings rather than a single binary
# file, so with copy_binary=False Phy has no raw data (dat_path = None): templates/features only.
# On a single-shank probe Phy can point at the saved binary directly with copy_binary=False.
# Set copy_binary=True to write a recording.dat for Phy's trace/raw-waveform views.
si.export_to_phy(analyzer, output_folder=path_to_data_folder / 'phy_KS4',
                compute_amplitudes=False, compute_pc_features=False, copy_binary=True,
                remove_if_exists=True, **job_kwargs)


# %% Reload from PHY after manual curation
sorting_phy_curated = si.read_phy(folder_path=path_to_data_folder / 'phy_KS4/', exclude_cluster_groups=['noise'])

# %%
"""
.##.....##.##....##.####.########....##.....##....###....########..######..##.....##
.##.....##.###...##..##.....##.......###...###...##.##......##....##....##.##.....##
.##.....##.####..##..##.....##.......####.####..##...##.....##....##.......##.....##
.##.....##.##.##.##..##.....##.......##.###.##.##.....##....##....##.......#########
.##.....##.##..####..##.....##.......##.....##.#########....##....##.......##.....##
.##.....##.##...###..##.....##.......##.....##.##.....##....##....##....##.##.....##
..#######..##....##.####....##.......##.....##.##.....##....##.....######..##.....##
"""

import UnitMatchPy.default_params as default_params
import UnitMatchPy.utils as util
import UnitMatchPy.overlord as ov
import UnitMatchPy.bayes_functions as bf
from DeepUnitMatch.utils import param_fun
from DeepUnitMatch.testing import test
import UnitMatchPy.assign_unique_id as aid
import UnitMatchPy.save_utils as su
import UnitMatchPy.metric_functions as mf
from DeepUnitMatch.utils import helpers

# ======================
# User inputs (edit me)
# ======================

# Paths to the Kilosort output directories (one per session)
KS_dirs = [r"Path\To\DeepUnitMatchExampleData\AL032\2019-11-21\Probe0\1", r"Path\To\DeepUnitMatchExampleData\AL032\2019-11-22\Probe0\1"]

# Where to save final results
save_dir = r"Path\To\Save\Dir"  # specify your save path here.
# Temporary folder for DeepUnitMatch preprocessed HDF5s (creates `processed_waveforms/` inside).
# This folder is cleared between runs — do not use it for anything you want to keep.
tmp_path = r"Path\To\Tmp\Dir"

# Model/inference settings
device = "cpu"

# Matching threshold (lower threshold -> more matches but also more false positives) - recommended 0.5
match_thresh = 0.5

# note, if the RawWaveforms does not yet exist, check extract_raw_data_demo.ipynb
# make sure RawWaveforms are of appropriate size when using our trained DeepUnitMatch model 
# (spike_width 82, samples_before 20, samples_after 61)

# %% Getting the data the same way as UnitMatch

# Get default parameters, can add your own before or after!
param = {}
param["KS_dirs"] = KS_dirs
param = default_params.get_default_param(param=param)
# param['spike_width'] = 61 # change if necessary!
# KS_dirs is defined in the "User inputs" cell above
wave_paths, unit_label_paths, channel_pos = util.paths_from_KS(
    KS_dirs
)  # note, if the RawWaveforms does not yet exist, check extract_raw_data_demo.ipynb
param = util.get_probe_geometry(channel_pos[0], param)

# STEP 0 from the UMPy example notebook
waveform, session_id, session_switch, within_session, good_units, param = (
    util.load_good_waveforms(wave_paths, unit_label_paths, param, good_units_only=True)
)
param["good_units"] = good_units

# %% Preprocess the DeepUnitMatch way and save as HDF5 files for each session in 'processed_waveforms'.

# tmp_path is defined in the "User inputs" cell above
unit_ids = np.concatenate(
    param["good_units"]
).squeeze()  # cluster IDs in the same order as `waveform`
snippets, positions, kept_idx = param_fun.get_snippets(
    waveform,
    channel_pos,
    session_id,
    save_path=tmp_path,
    unit_ids=unit_ids,
    param=param,
)

# If any units were rejected (bad waveform shape), re-sync all per-unit arrays so that
# waveform, session_id, session_switch and good_units all match the saved HDF5 files.
if len(kept_idx) < len(waveform):
    waveform, session_id, session_switch, within_session, good_units, param = (
        util.filter_units_by_index(
            waveform, session_id, session_switch, good_units, kept_idx, param
        )
    )

# Load the neural net
model = test.load_trained_model(device=device)

# We have stored the preprocessed data here (from the get_snippets function)
data_dir = os.path.join(tmp_path, "processed_waveforms")

# Pass the preprocessed data through the neural net
sim_matrix = test.inference(model, data_dir)

# Visualise the similarity matrix
plt.imshow(sim_matrix, cmap="viridis", aspect="auto")
plt.colorbar()

# %% Use the same Naive Bayes as in UnitMatchPy

clus_info = {
    "good_units": param["good_units"],
    "session_switch": session_switch,
    "session_id": session_id,
    "original_ids": np.concatenate(param["good_units"]),
}
extracted_wave_properties = ov.extract_parameters(
    waveform, channel_pos, clus_info, param
)  # contains spatial locations
within_session = 1 - (session_id[:, None] == session_id).astype(int)
sessions = np.unique(session_id)
match_dfs = []
probs = np.zeros(sim_matrix.shape)
distance_matrix = np.zeros(sim_matrix.shape)

# sessions is sorted (np.unique), so each pair comes once with r1 < r2
for r1, r2 in tqdm(list(combinations(sessions, 2)), desc="DeepUnitMatch (session pairs)"):

    mask = np.isin(session_id, [r1, r2])
    sim_mat = sim_matrix[mask][:, mask]
    n = np.sum(mask)
    n_units_r1 = session_switch[r1 + 1] - session_switch[r1]
    n_units_r2 = session_switch[r2 + 1] - session_switch[r2]
    session_switch_pair = np.array([0, n_units_r1, n_units_r1 + n_units_r2])

    indices = np.where(np.isin(session_id, [r1, r2]))[0]
    df = helpers.create_dataframe(
        [param["good_units"][r1], param["good_units"][r2]],
        sim_mat,
        session_list=[r1, r2],
    )
    matches = test.get_matches(
        df, sim_mat, session_id[indices], data_dir, dist_thresh=50
    )

    labels = np.eye(sim_mat.shape[0])
    subsessionid = np.array(
        [r1] * len(param["good_units"][r1]) + [r2] * len(param["good_units"][r2])
    )
    for (recses1, recses2), group in matches.groupby(by=["RecSes1", "RecSes2"]):
        asmatrix = (
            group["match"]
            .values.reshape(
                len(param["good_units"][recses1]), len(param["good_units"][recses2])
            )
            .astype(int)
        )
        labels[np.ix_(subsessionid == recses1, subsessionid == recses2)] = asmatrix

    # Visualize centroid distances for this session pair
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(labels, cmap="viridis", aspect="auto")
    plt.title("labels")
    plt.colorbar()
    plt.tight_layout()

    avg_centroid, avg_waveform_per_tp = (
        extracted_wave_properties["avg_centroid"][:, mask, :],
        extracted_wave_properties["avg_waveform_per_tp"][:, mask, :, :],
    )
    avg_waveform_per_tp = mf.drift_correct_session_pair(
        labels.astype(bool),
        session_switch_pair,
        avg_centroid,
        avg_waveform_per_tp,
        0,
        param,
    )
    avg_waveform_per_tp_flip = mf.flip_dim(avg_waveform_per_tp, param, n)
    euclid_dist = mf.get_Euclidean_dist(avg_waveform_per_tp_flip, param, n)
    centroid_dist, _ = mf.centroid_metrics(euclid_dist, param)
    # Visualize centroid distances for this session pair
    plt.subplot(1, 2, 2)
    plt.imshow(centroid_dist, cmap="viridis", aspect="auto")
    plt.title(f"Centroid dist: sessions {r1}-{r2}")
    plt.colorbar()
    plt.tight_layout()
    scores_to_incl = {
        "similarity": sim_mat,
        "distance": centroid_dist,
    }

    n_units = int(np.sqrt(len(df)))
    priors = np.array([1 - 2 / n_units, 2 / n_units])
    parameter_kernels = bf.get_parameter_kernels(
        scores_to_incl, labels, np.unique(labels), param
    )
    predictors = np.stack([scores for scores in scores_to_incl.values()], axis=2)
    probability = bf.apply_naive_bayes(
        parameter_kernels, priors, predictors, param, np.unique(labels)
    )
    prob_matrix = probability[:, 1].reshape(n_units, n_units)
    # Debug: verify shapes match before assignment
    target_shape = np.ix_(mask, mask)
    target_rows = np.where(mask)[0]
    if prob_matrix.shape != centroid_dist.shape:
        tqdm.write(
            f"  WARNING: Shape mismatch! prob_matrix={prob_matrix.shape} vs centroid_dist={centroid_dist.shape}"
        )
    probs[np.ix_(mask, mask)] = prob_matrix
    distance_matrix[np.ix_(mask, mask)] = centroid_dist

plt.imshow(probs, cmap="viridis", aspect="auto")
plt.colorbar()

util.evaluate_output(probs, param, within_session, session_switch, match_threshold=match_thresh)

# Process the output probability matrix to get final set of matches (across sessions)
# thresh is defined in the "User inputs" cell above
final_matches = test.directional_filter_matrix(probs, session_id, match_thresh)
plt.imshow(final_matches, cmap="viridis")
plt.colorbar()

# Divide final number of matches by 2 to account for double counting in the matrix
print(
    f" Found {np.sum(final_matches) // 2} matches in these sessions using the threshold of {match_thresh}."
)

# %% Check performance using the AUC. This tests the agreement between DeepUnitMatch matches and functional scores.
isicorr = test.ISI_correlations(param)
auc_isi = test.AUC(final_matches, isicorr, session_id)
print(f"AUC (ISI correlations):            {auc_isi:.3f}")

refpopcorr = test.refpop_correlations(param)
auc_refpop = test.AUC(final_matches, refpopcorr, session_id)
print(f"AUC (ref. pop. correlation):       {auc_refpop:.3f}")

frdiff = test.FR_diff(param)
auc_fr = test.AUC(
    final_matches, -frdiff, session_id
)  # negate: lower FRDiff = better match
print(f"AUC (firing rate difference):      {auc_fr:.3f}")

cvdiff = test.ISI_CV_diff(param)
auc_cv = test.AUC(
    final_matches, -cvdiff, session_id
)  # negate: lower CVDiff = better match
print(f"AUC (ISI CV difference):           {auc_cv:.3f}")

# Natural image correlations require specific stimulus data that isn't always available,
# so this AUC is optional and skipped gracefully if it can't be computed.
try:
    natimcorr = test.natim_correlations(param)
    reliability = np.diag(
        natimcorr
    )  # Extract the diagonal (self-correlations) as reliability scores
    natimcorr[reliability < 0.2, :] = np.nan  # Set rows with low reliability to NaN
    natimcorr[:, reliability < 0.2] = np.nan  # Set columns with low reliability to NaN
    auc_natim = test.AUC(final_matches, natimcorr, session_id)
    print(f"AUC (natim correlations):          {auc_natim:.3f}")
except Exception as e:
    print(f"AUC (natim correlations):          skipped ({e})")

# %% Finally, we can do tracking using the UnitMatch tracking function. This assigns unique IDs to the neurons that will persist across many sessions.

UIDs = aid.assign_unique_id(probs, param, clus_info)

# Save the results
# save_dir is defined in the "User inputs" cell above
amplitude = extracted_wave_properties["amplitude"]
spatial_decay = extracted_wave_properties["spatial_decay"]
avg_centroid = extracted_wave_properties["avg_centroid"]
avg_waveform = extracted_wave_properties["avg_waveform"]
avg_waveform_per_tp = extracted_wave_properties["avg_waveform_per_tp"]
wave_idx = extracted_wave_properties["good_wave_idxs"]
max_site = extracted_wave_properties["max_site"]
max_site_mean = extracted_wave_properties["max_site_mean"]
su.save_to_output(
    save_dir,
    {"distance": distance_matrix},
    np.argwhere(final_matches),
    probs,
    avg_centroid,
    avg_waveform,
    avg_waveform_per_tp,
    max_site,
    distance_matrix,
    final_matches,
    clus_info,
    param,
    UIDs=UIDs,
    matches_curated=None,
    save_match_table=True,
)