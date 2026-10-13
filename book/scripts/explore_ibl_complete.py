# Complete desktop-script version of the IBL walkthrough.
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader

# ONE = Open Neurophysiology Environment, the IBL data access layer.
# The public endpoint lets anyone download data with those public credentials.
one_kwargs = dict(
    base_url='https://openalyx.internationalbrainlab.org',
    username='intbrainlab',
    password='international',
    silent=True,
)
one = ONE(**one_kwargs)

# Probe insertion IDs (pids) from the IBL Benchmarks dataset — a curated set of high-quality recordings.
pids = [
    '1a276285-8b0e-4cc9-9f0a-a3a002978724',  # 00 - Benchmark PIDS start
    '1e104bf4-7a24-4624-a5b2-c2c8289c0de7',
    '6638cfb3-3831-4fc2-9327-194b76cf22e1',
    '749cb2b7-e57e-4453-a794-f6230e4d0226',
    'd7ec0892-0a6c-4f4f-9d8f-72083692af5c',
    'da8dfec1-d265-44e8-84ce-6ae9c109b8bd',
    'dab512bd-a02d-4c1f-8dbc-9155a163efc0',
    'dc7e9403-19f7-409f-9240-05ee57cb7aea',
    'e8f9fba4-d151-4b00-bee7-447f0f3e752c',
    'eebcaf65-7fa4-4118-869d-a084e84530e2',
    'fe380793-8035-414e-b000-09bfe5ece92a',  # Benchmark PIDS stop
]

pid = pids[-1]
# SpikeSortingLoader is a convenience object that bundles the probe geometry,
# spike-sorting results, and raw data access for a single insertion.
ss = SpikeSortingLoader(pid=pid, one=one)

# Optional full-recording downloads (leave False to stream snippets only).
DOWNLOAD_AP = False
DOWNLOAD_LFP = False
DOWNLOAD_ALL_PIDS = False

if DOWNLOAD_AP:
    print(ss.download_raw_electrophysiology(band='ap'))
if DOWNLOAD_LFP:
    print(ss.download_raw_electrophysiology(band='lf'))
if DOWNLOAD_ALL_PIDS:
    for probe_id in pids:
        loader = SpikeSortingLoader(pid=probe_id, one=one)
        print(probe_id, loader.download_raw_electrophysiology(band='ap'))

from viewephys.gui import create_app

app = create_app()

# Run prepare_ibl_loader.py first, in the same Python session.
import scipy.signal
import ibldsp.voltage
from viewephys.gui import viewephys
from iblatlas.atlas import BrainRegions

# get the histology information
channels = ss.load_channels()

# stream=True fetches only the requested chunk on-the-fly — no need to download the full recording.
# band='ap' selects the action-potential band (300 Hz high-pass, 30 kHz sample rate).
sr = ss.raw_electrophysiology(band='ap', stream=True)
sr_lf = ss.raw_electrophysiology(band='lf', stream=True)

# %%
# ---------------------------------------------------------------------------
# Load a short snippet of raw voltage data
# ---------------------------------------------------------------------------

T0 = 500  # start time in seconds — pick a quiet moment well into the recording

# sr[samples, channels] — slice to get 0.5 s of data.
# We transpose (.T) so the result is (n_channels, n_samples): the convention expected by ibldsp.
# The last channel is the sync/TTL channel; dropping it keeps only the 384 neural channels.
raw = sr[int(T0 * sr.fs):int((T0 + 0.5) * sr.fs), :-1].T

# %%
# ---------------------------------------------------------------------------
# Step 1 — Butterworth high-pass filter
# ---------------------------------------------------------------------------
# A 3rd-order Butterworth filter at 300 Hz removes slow LFP oscillations and
# DC drift while preserving the sharp transients caused by action potentials.
# sosfiltfilt applies the filter twice (forward + backward) for zero phase distortion.

butter_kwargs = {'N': 3, 'Wn': 300, 'btype': 'highpass', 'fs': sr.fs}
sos = scipy.signal.butter(**butter_kwargs, output='sos')
butt = scipy.signal.sosfiltfilt(sos, raw)

# ---------------------------------------------------------------------------
# Step 2 — Full IBL destriping pipeline
# ---------------------------------------------------------------------------
# ibldsp.voltage.destripe runs the complete IBL pre-processing chain:
#   • High-pass Butterworth (same as above)
#   • ADC sample-shift correction (Neuropixels channels are not sampled simultaneously)
#   • Spatial k-filter (removes stripe artefacts correlated across channels)
#
# Use the channel labels loaded for this insertion to identify bad channels.

destriped = ibldsp.voltage.destripe(raw, fs=sr.fs, h=sr.geometry,
                                    neuropixel_version=sr.major_version,
                                    channel_labels=channels['labels'])

# ---------------------------------------------------------------------------
# Display all three stages in viewephys for side-by-side comparison
# ---------------------------------------------------------------------------
# viewephys is an IBL interactive viewer: scroll with the mouse wheel,
# adjust gain with the viewer controls, and pan by clicking and dragging.
regions = BrainRegions()
eqcs = {}
eqcs['raw'] = viewephys(raw, sr.fs, channels=channels, title='Raw (unfiltered)', t0=T0, br=regions)
eqcs['butt'] = viewephys(butt, sr.fs, channels=channels, title='Butterworth high-pass 300 Hz', t0=T0, br=regions)
eqcs['destriped'] = viewephys(destriped, sr.fs, channels=channels, title='Destriped (butter + ADC + k-filter)', t0=T0, br=regions)

# Additional exercise: stream and display five seconds of raw LFP.
raw_lf = sr_lf[int(T0 * sr_lf.fs):int((T0 + 5) * sr_lf.fs), :-1].T
eqcs['raw_lf'] = viewephys(
    raw_lf, sr_lf.fs, channels=channels, title='Raw LFP', t0=T0, br=regions
)
print('LFP cache folder:', sr_lf.target_dir)

# Keep desktop windows responsive until they are closed.
app.exec()
