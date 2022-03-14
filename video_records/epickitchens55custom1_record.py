from .video_record import VideoRecord


class EpicKitchens55Custom1_VideoRecord(VideoRecord):
    def __init__(self, tup):
        self._index = str(tup[0])
        self._series = tup[1]

    @property
    def participant_id(self):
        return self._series['participant_id']

    @property
    def untrimmed_video_name(self):
        return self._series['video_id']

    @property
    def start_frame(self):
        return int(self._series['start_frame']) - 1

    @property
    def end_frame(self):
        return int(self._series['stop_frame']) - 2

    @property
    def fps(self):
        return {'RGB': 60,
                'Flow': 30,
                'Spec': 60,
                'HandBox': 60,
                'HandBoxMask': 60,
                }

    @property
    def num_frames(self):
        return {'RGB': self.end_frame - self.start_frame,
                'Flow': (self.end_frame - self.start_frame) / 2,
                'Spec': self.end_frame - self.start_frame,
                'HandBox': self.end_frame - self.start_frame,
                'HandBoxMask': self.end_frame - self.start_frame
                }
    @property
    def label(self):
        return {'verb': int(self._series['verb_class']) if 'verb_class' in self._series else -1,
                'noun': int(self._series['noun_class']) if 'noun_class' in self._series else -1}

    @property
    def metadata(self):
        return {
            'uid': self._index,
            'participant_id': self._series['participant_id'],
            'video_id': self._series['video_id'],
            'start_frame': self._series['start_frame'],
            'stop_frame': self._series['stop_frame']
        }
