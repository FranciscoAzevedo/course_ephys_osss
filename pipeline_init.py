# %% 
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import spikeinterface.full as si
import pandas as pd
from probeinterface.plotting import plot_probe
import spikeinterface.preprocessing as spre
import spikeinterface.widgets as sw
import spikeinterface.curation as sc
import seaborn as sns
from upsetplot import from_indicators, plot as plot_upset


n_cpus = os.cpu_count()
n_jobs = max(1, n_cpus - 4)
job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=True)

# This should be the path to the folder containing
# the data (.bin and .meta files linked above). 
path_to_data_folder = Path("/media/paco/Data/Prannays_ephys_test")

def read_spikeglx_ap(folder):
    # NP2.0 quad-base probes (NP2020) save 1536 AP + 4 sync channels (snsApLfSy=1536,0,4).
    # neo only strips the sync channel when there is exactly one, so get_traces() without
    # channel_ids returns all 1540 columns while the recording reports 1536 channels
    # (-> broadcast errors e.g. in detect_bad_channels). Selecting the channels explicitly
    # makes every get_traces() call request the 1536 AP channels only.
    rec = si.read_spikeglx(folder, stream_id="imec0.ap")
    return rec.select_channels(rec.channel_ids)

def detect_bad_channels_per_shank(rec, **kwargs):
    # coherence+psd compares each channel with its neighbours along depth. On a multi-shank
    # probe, ordering all channels by (x, y) interleaves the shanks, so run it per shank.
    # Labels are returned in the channel order of `rec`.
    bad_channel_ids = []
    channel_labels = np.empty(rec.get_num_channels(), dtype=object)
    for shank_rec in rec.split_by("group").values():
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

# 0.1 seconds of data from our raw recording
start_time = recording.get_times()[0]

si.plot_traces(
    recording,
    return_in_uV=True, # to ensure the data is returned in microvolts
    order_channel_by_depth=True
)
plt.show()

"""
TIP: In general when working with raw data, you can use recording.get_channel_locations()
to inspect the default channel ordering of the channels, and order_channels_by_depth() 
function can be used to get the indices for reordering.
"""

# %% Bad channels and cutting the recording for faster processing

# Checking for bad channels 
bad_channel_ids, channel_labels = detect_bad_channels_per_shank(recording, method='coherence+psd')

print('bad_channel_ids', bad_channel_ids)
print('channel_labels', channel_labels)
print('\n Percentage of bad channels:', len(bad_channel_ids) / recording.get_num_channels() * 100)

good_chn_recording = recording.remove_channels(bad_channel_ids)

# Cutting
# fs = recording.get_sampling_frequency()
# recording_sub = recording.frame_slice(start_frame=0*fs, end_frame=300*fs)
# recording_sub

# %% Pre-processing (phase_shift, bandpass_filter, common_reference)
# what about saturation detection, whitening?

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

# Splitting and plotting data by shank
split_recording = cmr_recording.split_by("group")
print(split_recording) # check if the groups match the ones defined in your geometry!

fig, ax = plt.subplots(1, len(split_recording.items()))

for shank_idx, shank_recording in split_recording.items():

    si.plot_traces(
        shank_recording,
        return_in_uV=True,
        order_channel_by_depth=True,
        ax=ax[shank_idx]
    )
    ax[shank_idx].set_title(f"Shank: {shank_idx}")

# Motion correction - DOESNT APPLY CLEANUP, JUST FOR VIZ
rec_corrected, motion_info = spre.correct_motion(
    shank_recording,
    preset="dredge_fast",
    output_motion_info=True,
    # turn off non-rigid estimation
    estimate_motion_kwargs={"rigid": True},  
)
print(rec_corrected)

# Display the motion output: drift map + estimated motion over depth/time.
fig = plt.figure(figsize=(14, 8))
sw.plot_motion_info(
    motion_info,
    recording=rec_corrected,
    figure=fig,
    color_amplitude=True,
    amplitude_cmap="inferno",
    scatter_decimate=10,
)
fig.suptitle("Motion correction output")

# %% Save preprocessed recording here
for shank_idx, shank_recording in split_recording.items():

    recording_saved = shank_recording.save(folder=path_to_data_folder / "preprocessed_" / f"shank_{shank_idx}", **job_kwargs)

# %% Sorting via Kilosort4 - ONLY FOR ONE SHANK ATM!
for shank_idx, shank_recording in split_recording.items():

    sorting = si.run_sorter(
        "kilosort4",
        shank_recording,
        folder=path_to_data_folder / f"kilosort4_output_shank{shank_idx}",
        remove_existing_folder=True,
        do_CAR=False,
        highpass_cutoff=0.1,
    )

    analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=shank_recording
    )

print(sorting)

# %% Alternative: load an existing sorting (Kilosort4 / IBL-sorter) instead of re-sorting
# Works on any Kilosort/Phy-style folder (spike_times.npy, spike_clusters.npy, params.py, ...),
# including one curated in Phy (merges/splits in spike_clusters.npy, labels in cluster_group.tsv).
# An ALF-converted IBL folder (spikes.samples.npy, clusters.*.npy) is also accepted.
# The sorter ran on the whole probe of the raw .bin, so spike times are sample indices
# into the full recording: pair them with all channels and no time slicing.
path_to_sorting_output = Path("/media/storage/shared-paton/Prannay/ephy_test/Behav_28-7_11268_day2_dopa/myRun22_g0/myRun22_g0_imec0")  # <- folder with params.py
path_to_raw_data = path_to_sorting_output  # <- folder with the .ap.bin/.ap.meta it was sorted from

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

loaded_recording = read_spikeglx_ap(path_to_raw_data)
sorting = load_existing_sorting(path_to_sorting_output, loaded_recording.get_sampling_frequency())
print(loaded_recording)
print(sorting)

# Preprocess the full probe like Kilosort4 did (highpass + CAR), interpolating bad
# channels instead of removing them so every unit keeps its peak channel
loaded_preprocessed = si.highpass_filter(loaded_recording, freq_min=300.0)
loaded_preprocessed = si.phase_shift(loaded_preprocessed)
loaded_bad_channel_ids, _ = detect_bad_channels_per_shank(loaded_preprocessed, method="coherence+psd")
print("bad_channel_ids", loaded_bad_channel_ids)
loaded_preprocessed = spre.interpolate_bad_channels(loaded_preprocessed, loaded_bad_channel_ids)

# Reference each shank separately, then re-join them into one recording
loaded_preprocessed = si.aggregate_channels(
    [si.common_reference(shank_rec, operator="median") for shank_rec in loaded_preprocessed.split_by("group").values()]
)

# Sanity checks: same sampling rate and spikes inside the recording
assert np.isclose(sorting.get_sampling_frequency(), loaded_preprocessed.get_sampling_frequency()), (
    f"Sampling frequencies differ: sorting {sorting.get_sampling_frequency()} Hz, "
    f"recording {loaded_preprocessed.get_sampling_frequency()} Hz"
)
last_spike_frame = sorting.to_spike_vector()["sample_index"].max()
assert last_spike_frame < loaded_preprocessed.get_num_samples(), (
    "Spikes fall after the end of the recording: is this the right .bin for this sorting?"
)

analyzer = si.create_sorting_analyzer(
    sorting=sorting,
    recording=loaded_preprocessed,
    sparse=True,
    **job_kwargs
)
print(analyzer)

# %% Quality control with Bombcell

extensions_dict = {
    'random_spikes': {},
    'templates': {},
    'waveforms': {},
    'noise_levels': {},
    'correlograms': {},
    'spike_amplitudes': {},
    'spike_locations': {},
    'unit_locations': {},
    'template_similarity': {},
    'principal_components': {},
    'amplitude_scalings': {},
    'isi_histograms': {},
    'quality_metrics': {},
    'template_metrics': {}
}

analyzer.compute(extensions_dict)

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


# %% EXPORT TO PHY
we = si.extract_waveforms(recording_saved, sorting, folder=path_to_data_folder / "waveforms_sparse", 
                          sparse=True, **job_kwargs)

si.export_to_phy(we, output_folder=path_to_data_folder / 'phy_KS4', 
                    compute_amplitudes=False, compute_pc_features=False, copy_binary=False,
                 **job_kwargs)


# %% Reload from PHY after manual curation
sorting_phy_curated = si.read_phy(folder_path=path_to_data_folder / 'phy_KS4/', exclude_cluster_groups=['noise'])

# %% Alternative: load data already curated
# NOT FUNCTIONAL: mismatch between labels imported and default params from Bombcell
# Better to load raw data and re-run Bombcell labeling with default thresholds
folder = Path("/home/paco/Desktop/ccu/code_data/course-ephys-osss/data/Mouse1/AL032/2019-11-21/Probe0/1/")

# Load the sorting output (handles Phy/Kilosort structure)
sorting = si.read_phy(folder)
# Alternatively: sorting = si.read_kilosort(folder_path)

# 1. Load precomputed Bombcell metrics
bc_df = pd.read_parquet(f"{folder}/templates._bc_qMetrics.parquet")

# 2. Check value counts of unit labels
# (Bombcell stores unit types in 'bc_unitType' or similar column)
if "bc_unitType" in bc_df.columns:
    print("Bombcell Label Counts:")
    print(bc_df["bc_unitType"].value_counts())
elif "unitType" in bc_df.columns:
    print(bc_df["unitType"].value_counts())

# 3. Plot Metric Histograms (recalculates sw.plot_metric_histograms)
# Select numerical metric columns to inspect
metric_cols = [
    col for col in bc_df.columns 
    if bc_df[col].dtype in ['float64', 'int64', 'float32']
][:12]  # plot first 12 metrics

fig, axes = plt.subplots(3, 4, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(metric_cols):
    sns.histplot(bc_df[col].dropna(), ax=axes[i], kde=True, color='skyblue')
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel('')

plt.tight_layout()
plt.suptitle("Precomputed Bombcell Metric Distributions", y=1.02, fontsize=14)
plt.show()

# Explicit mapping table for resolution
COLUMN_MAPPING = {
    'peak_to_trough_duration': ['peak_to_trough_duration'],
    'waveform_baseline_flatness': ['waveformBaselineFlatness'],
    'exp_decay': ['exp_decay'],
    'snr': ['signalToNoiseRatio'],
    'presence_ratio': ['presenceRatio'],
}

def find_column(df, si_key):
    for candidate in COLUMN_MAPPING.get(si_key, [si_key]):
        if candidate in df.columns:
            return candidate
    return None

def check_pass(val, rule):
    if pd.isna(val):
        return False
    if rule.get('abs', False):
        val = abs(val)
    if rule['greater'] is not None and val <= rule['greater']:
        return False
    if rule['less'] is not None and val >= rule['less']:
        return False
    return True

def classify_row(row, df, params):
    # 1. Noise check
    for key, rule in params['noise'].items():
        col = find_column(df, key)
        if col is None: continue
        val = row[col]
        # Auto-adjust if duration is saved in milliseconds instead of seconds
        if 'duration' in key and val > 0.1: 
            val /= 1000.0
        if not check_pass(val, rule):
            return 'noise'

    # 2. Non-somatic check
    for key, rule in params['non-somatic'].items():
        col = find_column(df, key)
        if col is None: continue
        val = row[col]
        if 'width' in key and val > 0.1: 
            val /= 1000.0
        if not check_pass(val, rule):
            return 'non-somatic'

    # 3. MUA check
    for key, rule in params['mua'].items():
        col = find_column(df, key)
        if col is None: continue
        val = row[col]
        # Adjust amplitude cutoff if expressed as percentage (0-100) instead of fraction (0-1)
        if key == 'amplitude_cutoff' and val > 1.0: 
            val /= 100.0
        if not check_pass(val, rule):
            return 'mua'

    return 'good'

# Apply classification to all cells
default_params = sc.bombcell_get_default_thresholds()
bc_df['bombcell_label'] = bc_df.apply(lambda row: classify_row(row, bc_df, default_params), axis=1)

# Summary output
print("--- Classification Results ---")
print(bc_df['bombcell_label'].value_counts())

# 4. Plot UpSet Plot for Rejection Criteria (recalculates sw.plot_bombcell_labels_upset)
# Bombcell boolean pass/fail columns usually start with 'use' or 'pass' or 'is'
boolean_cols = [col for col in bc_df.columns if bc_df[col].dtype == 'bool']

if boolean_cols:
    # Invert pass flags to highlight fail criteria for UpSet plot
    fail_df = ~bc_df[boolean_cols]
    
    # Generate UpSet plot
    upset_data = from_indicators(fail_df)
    plot_upset(upset_data, subset_size='count', show_counts=True)
    plt.suptitle("Bombcell Rejection Criteria Overlap", y=1.02)
    plt.show()