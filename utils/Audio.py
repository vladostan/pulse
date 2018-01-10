import pydub
import numpy as np

class Audio:
    def __init__(self, path=None, file_type=None, is_raw_data=True, data=None):
        self.is_raw_data = is_raw_data
        if self.is_raw_data:
            self.path = path
            self.file_type = file_type
        else:
            self.data = data
            self.raw = []

    def __enter__(self):
        if self.is_raw_data:
            if self.file_type == "ogg":
                self.data = pydub.AudioSegment.from_ogg(self.path)
                self.data = self.data.overlay(pydub.AudioSegment.from_file(self.path))
                self.raw = (np.fromstring(self.data._data, dtype="int16") + 0.5) / (0x7FFF + 0.5)
            else:
                print ("file type is not supported yet... add new file handler to this format..")
        return (self)

    def __exit__(self, exception_type, exception_value, traceback):
        del self.data
        del self.raw