
import mne
import os.path as op
from mne.datasets import sample

# print(__doc__)
#
# data_path = sample.data_path()
#
# fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
# evoked = mne.read_evokeds(fname, condition='Left Auditory',
#                           baseline=(None, 0))
#
# # plot with bads
# evoked.plot(exclude=[])
#
# # compute interpolation (also works with Raw and Epochs objects)
# evoked.interpolate_bads(reset_bads=False)
#
# # plot interpolated (previous bads)
# evoked.plot(exclude=[])

data_path = op.join(mne.datasets.sample.data_path(), 'MEG', 'sample')
raw = mne.io.read_raw_fif(op.join(data_path, 'sample_audvis_raw.fif'))
raw.set_eeg_reference()  # set EEG average reference
events = mne.read_events(op.join(data_path, 'sample_audvis_raw-eve.fif'))
