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

# CPU parallelism for SpikeInterface's chunked processing (saving, analyzer extensions...).
# These steps are numpy-only, so this does not affect the GPU: Kilosort4 runs on the GPU
# through torch_device in run_sorter below.
n_cpus = os.cpu_count()
n_jobs = max(1, n_cpus - 4)
job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=True)

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
path_to_data_folder = Path("D:\\tests_run\\G_first_try\\sofias_cable_animal_I_g0\\sofias_cable_animal_I_g0_imec0")

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
saved_corrected, motion_info = {}, {}
for shank_idx, shank_recording in split_recording.items():

    saved_corrected[shank_idx], motion_info[shank_idx] = spre.correct_motion(
    shank_recording,
    preset="dredge_fast",
    output_motion_info=True,
    # turn off non-rigid estimation
    estimate_motion_kwargs={"rigid": True},)

    print(saved_corrected[shank_idx])


# Display the motion output: drift map + estimated motion over depth/time ON ONE SHANK
fig = plt.figure(figsize=(14, 8))
sw.plot_motion_info(
    motion_info,
    recording=saved_corrected[shank_idx],
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
# CHANGE IN CASE YOU WANT TO SORT ALL OF THE SHANKS TOGETHER 
# (e.g. for a single shank probe, or if you want to allow units to span shanks)
sortings = {}
for shank_idx, shank_recording in saved_recordings.items():

    sortings[shank_idx] = si.run_sorter(
        "kilosort4",
        shank_recording,
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

# %% Analyzer for the Kilosort4 sorting: built once, saved to disk, reloaded afterwards
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
    # One analyzer for all shanks (units get unique ids, shank in the "aggregation_key" property).
    # Default radius sparsity (~25 channels per unit) keeps each unit on its own shank; the
    # binary_folder format keeps extensions on disk instead of RAM and saves them as they are computed.
    analyzer = si.create_sorting_analyzer(
        sorting=sortings,
        recording=saved_recordings,
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
path_to_sorting_output = Path("/media/paco/Data/Behav_2-8_11268_day7_sal_dls/myRun22_g0/myRun22_g0_imec0")  # <- folder with params.py
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
# shank of each channel from channel_groups.npy). The analyzer's recording is an aggregation
# of the per-shank saved recordings rather than a single binary file, so with
# copy_binary=False Phy has no raw data (dat_path = None): templates/features only.
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

for r1 in sessions:
    for r2 in sessions:
        if r1 >= r2:
            continue

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
            print(
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