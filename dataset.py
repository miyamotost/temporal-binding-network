from video_records import EpicKitchens55_VideoRecord, EpicKitchens100_VideoRecord, EpicKitchens55Custom1_VideoRecord
import torch
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

from torch.nn.utils.rnn import pad_sequence

class TBNDataSet(data.Dataset):
    def __init__(self, dataset, list_file,
                 new_length, modality, image_tmpl,
                 visual_path=None, audio_path=None, handbox_path=None, handboxmask_path=None, handtraj_path=None,
                 resampling_rate=44000, num_segments=3, transform=None, mode='train', use_audio_dict=True):
        self.dataset = dataset
        if audio_path is not None:
            if not use_audio_dict:
                self.audio_path = Path(audio_path)
            else:
                self.audio_path = pickle.load(open(audio_path, 'rb'))

        self.visual_path = visual_path
        self.handbox_path = handbox_path
        self.handboxmask_path = handboxmask_path
        self.handtraj_path = handtraj_path
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
                if record.untrimmed_video_name not in self.not_found_frames_video_id:
                    self.not_found_frames_video_id.append(record.untrimmed_video_name)
                imgs = [Image.new('L',(456,256)), Image.new('L',(456,256))]
            return imgs
        elif modality == 'Spec':
            spec = self._extract_sound_feature(record, idx)
            return [Image.fromarray(spec)]
        elif modality == 'HandBoxMask':
            idx_untrimmed = record.start_frame + idx
            handboxmask_dir = '{}/{}/{}_{}_{}'.format(self.handboxmask_path,
                                                    record.participant_id, record.untrimmed_video_name,
                                                    record.start_frame+1, record.end_frame+2)
            img_path = '{}/{}'.format(handboxmask_dir, 'frame_{:010d}_det.png'.format(idx_untrimmed))

            if os.path.exists(img_path):
                img = [Image.open(img_path).convert('RGB')]
            else:
                #print('Not found, RGB: {} {}'.format(record.untrimmed_video_name, self.image_tmpl[modality].format(idx_untrimmed)))
                if record.untrimmed_video_name not in self.not_found_frames_video_id:
                    self.not_found_frames_video_id.append(record.untrimmed_video_name)
                img = [Image.new('RGB',(456,256))]
                print('not found handboxmask {}_{}'.format(record.untrimmed_video_name, idx_untrimmed))
            return img

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

        #print('{} / {} / {}'.format(record.untrimmed_video_name, record.start_frame, record.end_frame))

        seq_length = -1
        for m in self.modality:
            if m == 'HandTraj':
                #no temporal binding (overall, start to stop)
                #[1 x consensus size, frames(veriable length), trajectry elements]
                #Ex1. [3, 2300, 2(x, y)], Ex2. [3, 2300, 4(x1, y1, x2, y2)]
                img, label, meta = self.get(m, record, None)
                input[m] = img
                #seq_length = meta['hand_traj_seq_length']
                continue

            if self.mode == 'train':
                segment_indices = self._sample_indices(record, m)
            elif self.mode == 'val':
                segment_indices = self._get_val_indices(record, m)
            elif self.mode == 'test':
                segment_indices = self._get_test_indices(record, m)

            # TBWを実現するために、モダリティごとに、複数のindexを用いたリストを作成する。
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

        if modality == 'HandBox': # HandBox
            handbox_fine_name = '{}_{}_{}.pkl'.format(record.untrimmed_video_name, record.start_frame+1, record.end_frame+2)
            handbox_file_path = '{}/{}/{}'.format(self.handbox_path, record.participant_id, handbox_fine_name)
            with open(handbox_file_path, 'rb') as f:
                handbox_data = pickle.load(f)
                handbox_indice = [int(d['frame_index'].replace('frame_', '').replace('.jpg', '')) for d in handbox_data]
                c_sum = 0
                c_not_found = 0
                for seg_ind in indices:
                    p = int(seg_ind)
                    for i in range(self.new_length[modality]):
                        c_sum += 1
                        id = record.start_frame + p

                        seg_imgs = None
                        for i, frame in enumerate(handbox_indice):
                            if frame == id:
                                seg_imgs = handbox_data[i]['hand_dets'] # shape = [count, data]
                                if seg_imgs is not None:
                                    seg_imgs = seg_imgs[np.newaxis, :, :]   # shape = [1, count, data]
                        if seg_imgs is None:
                            #print('{}: seg_imgs is None at index_{}. use zeros.'.format(handbox_fine_name, id))
                            c_not_found += 1
                            seg_imgs = np.zeros((1, 2, 10))

                        if seg_imgs.shape[1] == 1: # 手検出データが1つの場合は、空リストを使って2つにする。
                            zeros = np.zeros((1, 1, seg_imgs.shape[2])).astype(np.float32)
                            seg_imgs = np.concatenate([seg_imgs, zeros], 1)
                        if seg_imgs.shape[1] >= 3: # 手検出データが3つ以上の場合は、最初の2つだけ使う
                            seg_imgs = seg_imgs[:, :2, :]

                        images.extend(seg_imgs)
                        if p < record.num_frames[modality]:
                            p += 1
                #if c_not_found > 0:
                #    print('{}: not found hand_dets -> {}/{}.'.format(handbox_fine_name, c_not_found, c_sum))
            process_data = torch.from_numpy(np.array(images, dtype=np.float32))
        elif modality == 'HandTraj':
            filename = '{}_{}_{}.pkl'.format(record.untrimmed_video_name, record.start_frame+1, record.end_frame+2)
            handbox_file_path = '{}/{}'.format(self.handtraj_path, filename)
            max_length = 2000
            """
            if os.path.exists(handbox_file_path):
                with open(handbox_file_path, 'rb') as f:
                    handbox_data = pickle.load(f)
                    hand_L, hand_R = np.array(handbox_data['hand_L']), np.array(handbox_data['hand_R'])
                    if hand_L != [] and hand_R != []:
                        hand_L, hand_R = hand_L[np.newaxis, :, :], hand_R[np.newaxis, :, :]
                        hand_L, hand_R = hand_L[:, :, 4:6], hand_R[:, :, 4:6] # [center_x, center_y]
                        hand = np.concatenate([hand_L, hand_R], 2) # [1, frame, :]
                    else:
                        print('hand_L and hand_R are empty.')
                        hand = np.full((1, 1, 4), -1) # [1, frame, :]
                    hand = np.concatenate([hand, hand, hand], 0)

                    # same length with pad
                    # [consensus, sequent length, elements] -> [consensus, max width, elements]
                    try:
                        hand = np.pad(hand, [(0, 0), (0, max_length - hand.shape[1]), (0, 0)], "constant")
                    except ValueError:
                        print(hand.shape[1], max_length, max_length - hand.shape[1])
                        #exit()
                        hand = hand[:, :max_length, :]
            else:
                print('{} is not found.'.format(filename))
                hand = np.random.rand(3, max_length, 4)
            """

            """
            if os.path.exists(handbox_file_path):
                with open(handbox_file_path, 'rb') as f:
                    handbox_data = pickle.load(f)
                    hand_L, hand_R = np.array(handbox_data['hand_L']), np.array(handbox_data['hand_R'])
                    if hand_L != [] and hand_R != []:
                        hand_L, hand_R = hand_L[np.newaxis, :, :], hand_R[np.newaxis, :, :]
                        hand_L, hand_R = hand_L[:, :, 0:4], hand_R[:, :, 0:4] # [x,y,x,y]
                        hand = np.concatenate([hand_L, hand_R], 2) # [1, frame, 8]
                    else:
                        print('hand_L and hand_R are empty.')
                        hand = np.full((1, 1, 8), -2) # [1, frame, :]
                    hand = np.concatenate([hand, hand, hand], 0)

                    # same length with pad
                    # [consensus, sequent length, elements] -> [consensus, max width, elements]
                    try:
                        hand = np.pad(hand, [(0, 0), (0, max_length - hand.shape[1]), (0, 0)], "constant")
                    except ValueError:
                        print(hand.shape[1], max_length, max_length - hand.shape[1])
                        #exit()
                        hand = hand[:, :max_length, :]
            else:
                print('{} is not found.'.format(filename))
                hand = np.random.rand(3, max_length, 8)
            """

            seq_length = 0
            element_length = 12
            if os.path.exists(handbox_file_path):
                with open(handbox_file_path, 'rb') as f:
                    handbox_data = pickle.load(f)
                    hand_L, hand_R = np.array(handbox_data['hand_L']), np.array(handbox_data['hand_R'])
                    if hand_L != [] and hand_R != []:
                        hand_L, hand_R = hand_L[np.newaxis, :, :], hand_R[np.newaxis, :, :]
                        hand_L, hand_R = hand_L[:, :, 0:6], hand_R[:, :, 0:6] # [x,y,x,y, cx, cy]
                        hand = np.concatenate([hand_L, hand_R], 2) # [1, frame, 8]
                    else:
                        print('hand_L and hand_R are empty.')
                        hand = np.full((1, 1, element_length), -2) # [1, frame, :]
                    #hand = np.concatenate([hand, hand, hand], 0)

                    # same length with pad
                    # [consensus, sequent length, elements] -> [consensus, max width, elements]
                    """
                    custom padding
                    try:
                        seq_length = hand.shape[1]
                        hand = np.pad(hand, [(0, 0), (0, max_length - hand.shape[1]), (0, 0)], "constant", constant_values=-2)
                    except ValueError:
                        print(hand.shape[1], max_length, max_length - hand.shape[1])
                        #exit()
                        seq_length = max_length # 軌道が最大長を超えるとき
                        hand = hand[:, :max_length, :]
                    """
            else:
                print('{} is not found.'.format(filename))
                seq_length = 0 # 軌道が存在しない
                hand = np.full((1, 1, element_length), -2)

            #hand = hand.astype(np.float32)
            hand = torch.from_numpy(np.array(hand, dtype=np.float32))
            #hand = np.random.rand(3, 2000, 4)
            #hand = np.random.rand(3, 16000).astype(np.float32)
            #hand = torch.reshape(hand, (3, 8000)) # [3, 2000, 4] -> [3, 8000]
            #()()hand = torch.reshape(hand, (3, 16000))
            #hand = torch.reshape(hand, (3, 24000))
            #print(hand[:, -3:])
            process_data = hand
            #record.metadata['hand_traj_seq_length'] = seq_length
        else: # RGB, Flow, Spec, HandBoxMask
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
