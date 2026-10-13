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
