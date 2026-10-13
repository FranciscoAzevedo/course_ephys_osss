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
