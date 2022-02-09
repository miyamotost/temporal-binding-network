from video_records import EpicKitchens55_VideoRecord, EpicKitchens100_VideoRecord, EpicKitchens55Custom1_VideoRecord
import torch.utils.data as data

import librosa
from PIL import Image
import os
import os.path
from pathlib import Path
import pandas as pd
import numpy as np
from numpy.random import randint
import pickle


class TBNDataSet(data.Dataset):
    def __init__(self, dataset, list_file,
                 new_length, modality, image_tmpl,
                 visual_path=None, audio_path=None,
                 resampling_rate=44000,
                 num_segments=3, transform=None,
                 mode='train', use_audio_dict=True):
        self.dataset = dataset
        if audio_path is not None:
            if not use_audio_dict:
                self.audio_path = Path(audio_path)
            else:
                self.audio_path = pickle.load(open(audio_path, 'rb'))

        self.visual_path = visual_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.mode = mode
        self.resampling_rate = resampling_rate
        self.use_audio_dict = use_audio_dict

        self.not_found_frames_video_id = []

        if 'RGBDiff' in self.modality:
            self.new_length['RGBDiff'] += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _log_specgram(self, audio, window_size=10,
                     step_size=5, eps=1e-6):
        nperseg = int(round(window_size * self.resampling_rate / 1e3))
        noverlap = int(round(step_size * self.resampling_rate / 1e3))

        spec = librosa.stft(audio, n_fft=511,
                            window='hann',
                            hop_length=noverlap,
                            win_length=nperseg,
                            pad_mode='constant')

        spec = np.log(np.real(spec * np.conj(spec)) + eps)
        return spec

    def _extract_sound_feature(self, record, idx):

        centre_sec = (record.start_frame + idx) / record.fps['Spec']
        left_sec = centre_sec - 0.639
        right_sec = centre_sec + 0.639
        audio_fname = record.untrimmed_video_name + '.wav'
        if not self.use_audio_dict:
            samples, sr = librosa.core.load(self.audio_path / audio_fname,
                                            sr=None, mono=True)

        else:
            audio_fname = record.untrimmed_video_name
            samples = self.audio_path[audio_fname]

        duration = samples.shape[0] / float(self.resampling_rate)

        left_sample = int(round(left_sec * self.resampling_rate))
        right_sample = int(round(right_sec * self.resampling_rate))

        if left_sec < 0:
            samples = samples[:int(round(self.resampling_rate * 1.279))]

        elif right_sec > duration:
            samples = samples[-int(round(self.resampling_rate * 1.279)):]
        else:
            samples = samples[left_sample:right_sample]

        return self._log_specgram(samples)

    def _load_data(self, modality, record, idx):
        if modality == 'RGB' or modality == 'RGBDiff':
            idx_untrimmed = record.start_frame + idx
            img_path = os.path.join(self.visual_path, record.untrimmed_video_name, self.image_tmpl[modality].format(idx_untrimmed))
            if os.path.exists(img_path):
                img = [Image.open(img_path).convert('RGB')]
            else:
                #print('Not found, RGB: {} {}'.format(record.untrimmed_video_name, self.image_tmpl[modality].format(idx_untrimmed)))
                if record.untrimmed_video_name not in self.not_found_frames_video_id:
                    self.not_found_frames_video_id.append(record.untrimmed_video_name)
                img = [Image.new('RGB',(456,256))]
            return img
        elif modality == 'Flow':
            rgb2flow_fps_ratio = record.fps['Flow'] / float(record.fps['RGB'])
            idx_untrimmed = int(np.floor((record.start_frame * rgb2flow_fps_ratio))) + idx
            x_img_path = os.path.join(self.visual_path, record.untrimmed_video_name, self.image_tmpl[modality].format('x', idx_untrimmed))
            y_img_path = os.path.join(self.visual_path, record.untrimmed_video_name, self.image_tmpl[modality].format('y', idx_untrimmed))
            if (os.path.exists(x_img_path) and os.path.exists(y_img_path)):
                x_img = Image.open(x_img_path).convert('L')
                y_img = Image.open(y_img_path).convert('L')
                imgs = [x_img, y_img]
            else:
                #print('Not found, Flow: {} or {}'.format(self.image_tmpl[modality].format('x', idx_untrimmed), self.image_tmpl[modality].format('y', idx_untrimmed)))
                if record.untrimmed_video_name not in self.not_found_frames_video_id:
                    self.not_found_frames_video_id.append(record.untrimmed_video_name)
                imgs = [Image.new('L',(456,256)), Image.new('L',(456,256))]
            return imgs
        elif modality == 'Spec':
            spec = self._extract_sound_feature(record, idx)
            return [Image.fromarray(spec)]

    def _parse_list(self):
        if self.dataset == 'epic-kitchens-55':
            self.video_list = [EpicKitchens55_VideoRecord(tup) for tup in self.list_file.iterrows()]
        elif self.dataset == 'epic-kitchens-100':
            self.video_list = [EpicKitchens100_VideoRecord(tup) for tup in self.list_file.iterrows()]
        elif self.dataset == 'epic-kitchens-55-custom-1':
            self.video_list = [EpicKitchens55Custom1_VideoRecord(tup) for tup in self.list_file.iterrows() if tup[1]['mode'] == self.mode]
        print('dataset[{}] num={}'.format(self.mode, len(self.video_list)))

    def _sample_indices(self, record, modality):
        """

        :param record: VideoRecord
        :return: list
        """
        average_duration = (record.num_frames[modality] - self.new_length[modality] + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        # elif record.num_frames[modality] > self.num_segments:
        #     offsets = np.sort(randint(record.num_frames[modality] - self.new_length[modality] + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_val_indices(self, record, modality):
        if record.num_frames[modality] > self.num_segments + self.new_length[modality] - 1:
            tick = (record.num_frames[modality] - self.new_length[modality] + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_indices(self, record, modality):

        tick = (record.num_frames[modality] - self.new_length[modality] + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets

    def __getitem__(self, index):

        input = {}
        record = self.video_list[index]

        ###tmp
        #print('{} / {} / {}'.format(record.untrimmed_video_name, record.start_frame, record.end_frame))

        for m in self.modality:
            if self.mode == 'train':
                segment_indices = self._sample_indices(record, m)
            elif self.mode == 'val':
                segment_indices = self._get_val_indices(record, m)
            elif self.mode == 'test':
                segment_indices = self._get_test_indices(record, m)

            # We implement a Temporal Binding Window (TBW) with size same as the action's length by:
            #   1. Selecting different random indices (timestamps) for each modality within segments
            #      (this is similar to using a TBW with size same as the segment's size)
            #   2. Shuffling randomly the segments of Flow, Audio (RGB is the anchor hence not shuffled)
            #      which binds data across segments, hence making the TBW same in size as the action.
            #   Example of an action with 90 frames across all modalities:
            #    1. Synchronous selection of indices per segment:
            #       RGB: [12, 41, 80], Flow: [12, 41, 80], Audio: [12, 41, 80]
            #    2. Asynchronous selection of indices per segment:
            #       RGB: [12, 41, 80], Flow: [9, 55, 88], Audio: [20, 33, 67]
            #    3. Asynchronous selection of indices per action:
            #       RGB: [12, 41, 80], Flow: [88, 55, 9], Audio: [67, 20, 33]

            if m != 'RGB' and self.mode == 'train':
                np.random.shuffle(segment_indices)

            img, label, meta = self.get(m, record, segment_indices)
            input[m] = img

        meta['not_found_frames_video_id'] = self.not_found_frames_video_id

        return input, label, meta

    def get(self, modality, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length[modality]):
                seg_imgs = self._load_data(modality, record, p)
                images.extend(seg_imgs)
                if p < record.num_frames[modality]:
                    p += 1

        process_data = self.transform[modality](images)
        return process_data, record.label, record.metadata

    def __len__(self):
        return len(self.video_list)
