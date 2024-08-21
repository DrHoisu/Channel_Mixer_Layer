import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
try:
    import tensorflow as tf
except ImportError:
    tf = None

from openstl.datasets.utils import create_loader


class IONODataset(Dataset):
    """
    Args:
        datas, indices (list): Data and indices of path.
        image_size (int: The target resolution of GIM images.
        pre_seq_length (int): The input sequence length.
        aft_seq_length (int): The output sequence length for prediction.
        use_augment (bool): Whether to use augmentations (defaults to False).
    """

    def __init__(self, datas, indices, image_size=72, pre_seq_length=12, aft_seq_length=48, use_augment=False, aux_channel=0, exp_stage=None):
        super(IONODataset,self).__init__()
        self.datas = datas
        self.indices = indices
        self.image_size = image_size
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.tot_seq_length = self.pre_seq_length + self.aft_seq_length
        self.use_augment = use_augment
        self.mean = 0
        self.std = 1
        self.aux_channel = aux_channel
        self.exp_stage = exp_stage

    def _augment_seq(self, imgs, crop_scale=0.95):
        """Augmentations for video"""
        _, _, h, w = imgs.shape  # original shape, e.g., [10, 3, 64, 64]
        imgs = F.interpolate(imgs, scale_factor=1 / crop_scale, mode='bilinear')
        _, _, ih, iw = imgs.shape
        # Random Crop
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        imgs = imgs[:, :, x:x+h, y:y+w]
        # Random Flip
        if random.randint(0, 1):
            imgs = torch.flip(imgs, dims=(3, ))  # horizontal flip
        return imgs

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        batch_ind = self.indices[i]
        begin = batch_ind

        input_batch = np.zeros(
            (self.pre_seq_length + self.aft_seq_length, self.image_size, self.image_size, 20)).astype(np.float32)
        begin = batch_ind[-1]
        end = begin + self.tot_seq_length
        k = 0
        for serialized_example in tf.compat.v1.python_io.tf_record_iterator(batch_ind[0]):
            if k == batch_ind[1]:
                example = tf.train.Example()
                example.ParseFromString(serialized_example)
                break
            k += 1
        arrange = np.arange(72)
        longitude = arrange.reshape(1,72)
        longitude = np.repeat(longitude, repeats=72, axis=0)/72
        longitude = longitude[:,:,np.newaxis]
        latitude = arrange.reshape(72,1)
        latitude = np.repeat(latitude, repeats=72, axis=1)/72
        latitude = latitude[:,:,np.newaxis]
        solar_ratio = 1.
        for j in range(begin, end):
            image_name = str(j) + '/Map'
            image_value = example.features.feature[image_name].bytes_list.value[0]
            image_value = np.minimum(np.frombuffer(image_value, dtype=np.uint16), 2000) #* solar_ratio
            input_batch[j - begin, :, :, 0:1] = image_value.reshape((self.image_size, self.image_size, 1)) / 2000

            input_batch[j - begin, :, :, 1:2] = longitude

            input_batch[j - begin, :, :, 2:3] = latitude

            year_name = str(j) + '/Year'
            year_value = np.array(example.features.feature[year_name].float_list.value) / 2500
            input_batch[j - begin, :, :, 3:4] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in year_value], axis=2)

            F107_name = str(j) + '/F107'
            F107_value = np.array(example.features.feature[F107_name].float_list.value) * solar_ratio / 400
            input_batch[j - begin, :, :, 4:5] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in F107_value], axis=2)

            R_name = str(j) + '/R'
            R_value = np.array(example.features.feature[R_name].float_list.value) * solar_ratio / 400
            input_batch[j - begin, :, :, 5:6] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in R_value], axis=2)

            Day_name = str(j) + '/Day'
            Day_value = np.array(example.features.feature[Day_name].float_list.value) / 367
            input_batch[j - begin, :, :, 6:7] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in Day_value], axis=2)

            hour_name = str(j) + '/Hour'
            hour_value = np.array(example.features.feature[hour_name].float_list.value) / 24
            input_batch[j - begin, :, :, 7:8] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in hour_value], axis=2)

            month_name = str(j) + '/Month'
            month_value = np.array(example.features.feature[month_name].float_list.value) / 13
            input_batch[j - begin, :, :, 8:9] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in month_value], axis=2)

            date_name = str(j) + '/Date'
            date_value = np.array(example.features.feature[date_name].float_list.value) / 32
            input_batch[j - begin, :, :, 9:10] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in date_value], axis=2)

            Kp_name = str(j) + '/Kp'
            Kp_value = np.array(example.features.feature[Kp_name].float_list.value) / 100
            input_batch[j - begin, :, :, 10:11] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in Kp_value], axis=2)

            Ap_name = str(j) + '/Ap'
            Ap_value = np.array(example.features.feature[Ap_name].float_list.value) / 400
            input_batch[j - begin, :, :, 11:12] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in Ap_value], axis=2)

            Dst_name = str(j) + '/Dst'
            Dst_value = (np.array(example.features.feature[Dst_name].float_list.value) + 500) / 600
            input_batch[j - begin, :, :, 12:13] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in Dst_value], axis=2)

            PlasmaSpeed_name = str(j) + '/PlasmaSpeed'
            PlasmaSpeed_value = np.array(example.features.feature[PlasmaSpeed_name].float_list.value) / 1200
            input_batch[j - begin, :, :, 13:14] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in PlasmaSpeed_value], axis=2)

            FieldMagnitude_name = str(j) + '/FieldMagnitude'
            FieldMagnitude_value = np.array(example.features.feature[FieldMagnitude_name].float_list.value) / 70
            input_batch[j - begin, :, :, 14:15] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in FieldMagnitude_value], axis=2)

            SigmaB_name = str(j) + '/SigmaB'
            SigmaB_value = np.array(example.features.feature[SigmaB_name].float_list.value) / 50
            input_batch[j - begin, :, :, 15:16] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in SigmaB_value], axis=2)

            ProtonDensity_name = str(j) + '/ProtonDensity'
            ProtonDensity_value = np.array(example.features.feature[ProtonDensity_name].float_list.value) / 140
            input_batch[j - begin, :, :, 16:17] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in ProtonDensity_value], axis=2)

            FlowPressure_name = str(j) + '/FlowPressure'
            FlowPressure_value = np.array(example.features.feature[FlowPressure_name].float_list.value) / 80
            input_batch[j - begin, :, :, 17:18] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in FlowPressure_value], axis=2)

            ElectricField_name = str(j) + '/ElectricField'
            ElectricField_value = (np.array(example.features.feature[ElectricField_name].float_list.value) + 50) / 100
            input_batch[j - begin, :, :, 18:19] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in ElectricField_value], axis=2)

            MagnetosonicMach_name = str(j) + '/MagnetosonicMach'
            MagnetosonicMach_value = np.array(example.features.feature[MagnetosonicMach_name].float_list.value) / 15
            input_batch[j - begin, :, :, 19:20] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in MagnetosonicMach_value], axis=2)

        input_batch = input_batch[:, :, :, :(1+self.aux_channel)]
        input_batch = torch.tensor(input_batch).float().permute(0, 3, 1, 2)
        data = input_batch #[:self.pre_seq_length, ::]
        labels = input_batch[self.pre_seq_length:self.tot_seq_length, ::(1+self.aux_channel)]
        if self.use_augment:
            #imgs = self._augment_seq(torch.cat([data, labels], dim=0), crop_scale=0.95)
            imgs = self._augment_seq(data, crop_scale=0.95)
            data = imgs #[:self.pre_seq_length, ...]
            labels = imgs[self.pre_seq_length:self.tot_seq_length, ...]


        return data, labels


class InputHandle(object):
    """Class for handling dataset inputs."""

    def __init__(self, datas, indices, configs):
        self.name = configs['name']
        self.minibatch_size = configs['minibatch_size']
        self.image_height = configs['image_height']
        self.image_width = configs['image_width']
        self.datas = datas
        self.indices = indices
        self.current_position = 0
        self.current_batch_indices = []
        self.current_input_length = configs['seq_length']

    def total(self):
        return len(self.indices)

    def begin(self, do_shuffle=True):
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.minibatch_size]

    def next(self):
        self.current_position += self.minibatch_size
        if self.no_batch_left():
            return None
        else:
            self.current_batch_indices = self.indices[
                self.current_position:self.current_position + self.minibatch_size]

    def no_batch_left(self):
        return self.current_position + self.minibatch_size >= self.total()

    def get_batch(self):
        if self.no_batch_left():
            return None
        input_batch = np.zeros(
            (self.minibatch_size, self.current_input_length, self.image_height, self.image_width, 20)).astype(np.float32)

        arrange = np.arange(72)
        longitude = arrange.reshape(1,72)
        longitude = np.repeat(longitude, repeats=72, axis=0)/72
        longitude = longitude[:,:,np.newaxis]
        latitude = arrange.reshape(72,1)
        latitude = np.repeat(latitude, repeats=72, axis=1)/72
        latitude = latitude[:,:,np.newaxis]

        for i in range(self.minibatch_size):
            batch_ind = self.current_batch_indices[i]
            begin = batch_ind[-1]
            end = begin + self.current_input_length
            k = 0
            for serialized_example in tf.compat.v1.python_io.tf_record_iterator(batch_ind[0]):
                if k == batch_ind[1]:
                    example = tf.train.Example()
                    example.ParseFromString(serialized_example)
                    break
                k += 1
            for j in range(begin, end):
            	image_name = str(j) + '/Map'
            	image_value = example.features.feature[image_name].bytes_list.value[0]
            	image_value = np.minimum(np.frombuffer(image_value, dtype=np.uint16), 2000) #* solar_ratio
            	input_batch[i, j - begin, :, :, 0:1] = image_value.reshape((self.image_size, self.image_size, 1)) / 2000

            	input_batch[i, j - begin, :, :, 1:2] = longitude

            	input_batch[i, j - begin, :, :, 2:3] = latitude

            	year_name = str(j) + '/Year'
            	year_value = np.array(example.features.feature[year_name].float_list.value) / 2500
            	input_batch[i, j - begin, :, :, 3:4] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in year_value], axis=2)

            	F107_name = str(j) + '/F107'
            	F107_value = np.array(example.features.feature[F107_name].float_list.value) * solar_ratio / 400
            	input_batch[i, j - begin, :, :, 4:5] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in F107_value], axis=2)

            	R_name = str(j) + '/R'
            	R_value = np.array(example.features.feature[R_name].float_list.value) * solar_ratio / 400
            	input_batch[i, j - begin, :, :, 5:6] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in R_value], axis=2)

            	Day_name = str(j) + '/Day'
            	Day_value = np.array(example.features.feature[Day_name].float_list.value) / 367
            	input_batch[i, j - begin, :, :, 6:7] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in Day_value], axis=2)

            	hour_name = str(j) + '/Hour'
            	hour_value = np.array(example.features.feature[hour_name].float_list.value) / 24
            	input_batch[i, j - begin, :, :, 7:8] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in hour_value], axis=2)

            	month_name = str(j) + '/Month'
            	month_value = np.array(example.features.feature[month_name].float_list.value) / 13
            	input_batch[i, j - begin, :, :, 8:9] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in month_value], axis=2)

            	date_name = str(j) + '/Date'
            	date_value = np.array(example.features.feature[date_name].float_list.value) / 32
            	input_batch[i, j - begin, :, :, 9:10] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in date_value], axis=2)

            	Kp_name = str(j) + '/Kp'
            	Kp_value = np.array(example.features.feature[Kp_name].float_list.value) / 100
            	input_batch[i, j - begin, :, :, 10:11] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in Kp_value], axis=2)

            	Ap_name = str(j) + '/Ap'
            	Ap_value = np.array(example.features.feature[Ap_name].float_list.value) / 400
            	input_batch[i, j - begin, :, :, 11:12] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in Ap_value], axis=2)

            	Dst_name = str(j) + '/Dst'
            	Dst_value = (np.array(example.features.feature[Dst_name].float_list.value) + 500) / 600
            	input_batch[i, j - begin, :, :, 12:13] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in Dst_value], axis=2)

            	PlasmaSpeed_name = str(j) + '/PlasmaSpeed'
            	PlasmaSpeed_value = np.array(example.features.feature[PlasmaSpeed_name].float_list.value) / 1200
            	input_batch[i, j - begin, :, :, 13:14] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in PlasmaSpeed_value], axis=2)

            	FieldMagnitude_name = str(j) + '/FieldMagnitude'
            	FieldMagnitude_value = np.array(example.features.feature[FieldMagnitude_name].float_list.value) / 70
            	input_batch[i, j - begin, :, :, 14:15] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in FieldMagnitude_value], axis=2)

            	SigmaB_name = str(j) + '/SigmaB'
            	SigmaB_value = np.array(example.features.feature[SigmaB_name].float_list.value) / 50
            	input_batch[i, j - begin, :, :, 15:16] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in SigmaB_value], axis=2)

            	ProtonDensity_name = str(j) + '/ProtonDensity'
            	ProtonDensity_value = np.array(example.features.feature[ProtonDensity_name].float_list.value) / 140
            	input_batch[i, j - begin, :, :, 16:17] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in ProtonDensity_value], axis=2)

            	FlowPressure_name = str(j) + '/FlowPressure'
            	FlowPressure_value = np.array(example.features.feature[FlowPressure_name].float_list.value) / 80
            	input_batch[i, j - begin, :, :, 17:18] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in FlowPressure_value], axis=2)

            	ElectricField_name = str(j) + '/ElectricField'
            	ElectricField_value = (np.array(example.features.feature[ElectricField_name].float_list.value) + 50) / 100
            	input_batch[i, j - begin, :, :, 18:19] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in ElectricField_value], axis=2)

            	MagnetosonicMach_name = str(j) + '/MagnetosonicMach'
            	MagnetosonicMach_value = np.array(example.features.feature[MagnetosonicMach_name].float_list.value) / 15
            	input_batch[i, j - begin, :, :, 19:20] = np.stack([np.ones([self.image_size, self.image_size]) * i for i in MagnetosonicMach_value], axis=2)

        input_batch = input_batch.astype(np.float32)
        return input_batch


class DataProcess(object):
    """Class for preprocessing dataset inputs."""

    def __init__(self, configs):
        self.configs = configs
        self.data_path = configs['data_path']
        self.image_height = configs['image_height']
        self.image_width = configs['image_width']
        self.seq_len = configs['seq_length']

    def load_data(self, path, mode='train'):
        """Loads the dataset.
        Args:
            path: action_path.
            mode: Training or testing.
        Returns:
            A dataset and indices of the sequence.
        """
        assert mode in ['train', 'test']
        if mode == 'train':
            path = os.path.join(path, 'train')
        else:
            path = os.path.join(path, 'test')
        print('begin load data' + str(path))

        video_fullpaths = []
        indices = []

        tfrecords = os.listdir(path)
        tfrecords.sort()
        num_pictures = 0
        assert tf is not None and 'Please install tensorflow, e.g., pip install tensorflow'

        for tfrecord in tfrecords:
            filepath = os.path.join(path, tfrecord)
            video_fullpaths.append(filepath)
            k = 0
            for serialized_example in tf.compat.v1.python_io.tf_record_iterator(os.path.join(path, tfrecord)):
                example = tf.train.Example()
                example.ParseFromString(serialized_example)
                i = 0
                while True:
                    year_name = str(i) + '/Year'
                    year_value = np.array(example.features.feature[year_name].float_list.value)
                    if year_value.shape == (0,):  # End of frames/data
                        break
                    i += 1
                num_pictures += i
                for j in range(i - self.seq_len + 1):
                    indices.append((filepath, k, j))
                k += 1
        print("there are " + str(num_pictures) + " pictures")
        print("there are " + str(len(indices)) + " sequences")
        return video_fullpaths, indices

    def get_train_input_handle(self):
        train_data, train_indices = self.load_data(self.data_path, mode='train')
        return InputHandle(train_data, train_indices, self.configs)

    def get_test_input_handle(self):
        test_data, test_indices = self.load_data(self.data_path, mode='test')
        return InputHandle(test_data, test_indices, self.configs)


def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=12, aft_seq_length=48, in_shape=[72, 72],
              distributed=False, use_augment=False, use_prefetcher=False,
              drop_last=False, exp_stage=None, aux_channel=0):

    img_height = in_shape[-2] if in_shape is not None else 72
    img_width = in_shape[-1] if in_shape is not None else 72
    input_param = {
        'data_path': os.path.join(data_root, 'iono_electron'),
        'image_height': img_height,
        'image_width': img_width,
        'minibatch_size': batch_size,
        'seq_length': (pre_seq_length + aft_seq_length),
        'input_data_type': 'float32',
        'name': 'iono'
    }
    input_handle = DataProcess(input_param)
    train_input_handle = input_handle.get_train_input_handle()
    test_input_handle = input_handle.get_test_input_handle()

    train_set = IONODataset(train_input_handle.datas,
                            train_input_handle.indices,
                            img_height,
                            pre_seq_length, aft_seq_length,
                            use_augment=use_augment,
                            aux_channel=aux_channel,
                            exp_stage=exp_stage)
    test_set = IONODataset(test_input_handle.datas,
                           test_input_handle.indices,
                           img_height,
                           pre_seq_length, aft_seq_length,
                           use_augment=False,
                           aux_channel=aux_channel,
                           exp_stage=exp_stage)

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_vali = None
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_vali, dataloader_test


if __name__ == '__main__':
    dataloader_train, _, dataloader_test = \
        load_data(batch_size=16,
                  val_batch_size=4,
                  data_root='../../data/',
                  num_workers=4,
                  pre_seq_length=12, aft_seq_length=48, aux_channel=3)

    print(len(dataloader_train), len(dataloader_test))
    for item in dataloader_train:
        print(item[0].shape, item[1].shape)
        break
    for item in dataloader_test:
        print(item[0].shape, item[1].shape)
        break

