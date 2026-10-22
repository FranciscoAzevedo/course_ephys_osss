# %% 
# General imports
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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

n_cpus = os.cpu_count()
n_jobs = max(1, n_cpus - 4)
job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=True)


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

# Save preprocessed recording here
# One saved recording per shank, keyed by shank index (same keys as split_recording)
saved_recordings = {}
for shank_idx, shank_recording in split_recording.items():

    saved_recordings[shank_idx] = shank_recording.save(folder=path_to_data_folder / "preprocessed_" / f"shank_{shank_idx}", **job_kwargs, overwrite=True)

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
# (so the preprocessing chain is not recomputed on the fly)
sortings = {}
for shank_idx, shank_recording in saved_recordings.items():

    sortings[shank_idx] = si.run_sorter(
        "kilosort4",
        shank_recording,
        clear_cache=True,
        folder=path_to_data_folder / f"kilosort4_output_shank{shank_idx}",
        batch_size=30000,
        remove_existing_folder=True,
        do_CAR=False,
        highpass_cutoff=0.1,
    )

# one analyzer, a sorting per shank
analyzer = si.create_sorting_analyzer(
    sorting=sortings,
    recording=saved_recordings,
    set_sparsity_by_dict_key=True, # each unit's waveforms restricted to its own shank
    **job_kwargs,
)

for shank_idx, sorting in sortings.items():
    print(f"Shank {shank_idx}: {sorting}")

# if you need to have the analyzer on only one shank, you can do:
# keys = analyzer.get_sorting_property("aggregation_key")
# analyzer_shank0 = analyzer.select_units(analyzer.unit_ids[keys == 0])

# %% Alternative: load an existing sorting (Kilosort4 / IBL-sorter) instead of re-sorting
# Works on any Kilosort/Phy-style folder (spike_times.npy, spike_clusters.npy, params.py, ...),
# An ALF-converted IBL folder (spikes.samples.npy, clusters.*.npy) is also accepted.
path_to_sorting_output = Path("/media/paco/Data/Behav_28-7_11268_day2_dopa/myRun22_g0/myRun22_g0_imec0/")  # <- folder with params.py
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
    'waveforms': {},
    'noise_levels': {},
    'correlograms': {},
    'spike_amplitudes': {},
    'spike_locations': {},
    'unit_locations': {},
    'template_similarity': {},
    'amplitude_scalings': {},
    'isi_histograms': {},
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
# shank of each channel from channel_groups.npy). The analyzer's recording is an aggregation
# of the per-shank saved recordings rather than a single binary file, so with
# copy_binary=False Phy has no raw data (dat_path = None): templates/features only.
# Set copy_binary=True to write a recording.dat for Phy's trace/raw-waveform views.
si.export_to_phy(analyzer, output_folder=path_to_data_folder / 'phy_KS4',
                compute_amplitudes=False, compute_pc_features=False, copy_binary=True,
                remove_if_exists=True, **job_kwargs)


# %% Reload from PHY after manual curation
sorting_phy_curated = si.read_phy(folder_path=path_to_data_folder / 'phy_KS4/', exclude_cluster_groups=['noise'])
