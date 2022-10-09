import random

import tensorflow as tf
import tensorflow_addons as tfa


class FacesDataset:
    """Represent faces dataset."""

    def __init__(self, image_height, image_width, faces_count, train_data_path, test_data_path) -> None:
        self.image_height = image_height
        self.image_width = image_width
        self.faces_count = faces_count
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

    @staticmethod
    def parse_tfrecord_fn():
        def _parse_tfrecord_fn(example):
            feature_description = {
                "image": tf.io.FixedLenFeature([], tf.string),
                "bboxes": tf.io.VarLenFeature(tf.float32),
                "faces_count": tf.io.FixedLenFeature([], tf.int64)
            }
            example = tf.io.parse_single_example(example, feature_description)
            example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
            example["bboxes"] = tf.sparse.to_dense(example["bboxes"])
            example['faces_count'] = example['faces_count']
            return example

        return _parse_tfrecord_fn

    @staticmethod
    def filter_amount_faces(faces_count):
        def _filter_amount_faces(features):
            return features['faces_count'] <= faces_count

        return _filter_amount_faces

    def _prepare_sample(self, fc):
        def prepare_sample(features):
            faces_count = features['faces_count']
            image = features["image"]
            image_shape = tf.shape(image)
            scaleY = 1 / image_shape[0]
            scaleX = 1 / image_shape[1]
            bboxes = features['bboxes']
            bboxes = tf.multiply(bboxes, tf.cast(tf.repeat([[scaleX, scaleY, scaleX, scaleY]], [faces_count]),
                                                 dtype=tf.float32))
            image = tf.image.resize(features["image"], size=(self.image_width, self.image_height)) / 255
            image = tf.image.random_contrast(image, 0.9, 1)
            reshaped = bboxes[0:4 * fc]

            h_shift = 50.0 * tf.cast(tf.random.uniform(shape=[], minval=-1, maxval=1), dtype=tf.float64)
            v_shift = 50.0 * tf.cast(tf.random.uniform(shape=[], minval=-1, maxval=1), dtype=tf.float64)
            image = tfa.image.translate(image, [h_shift, v_shift],
                                        interpolation='nearest', fill_mode='nearest')
            reshaped = reshaped + [h_shift / self.image_width, v_shift / self.image_height, 0.0, 0.0]

            mirror_needed = random.random() > 0.5
            if mirror_needed:
                image = tf.image.flip_left_right(image)
                reshaped = tf.abs([1.0 - reshaped[2], 0.0, 0.0, 0.0] - reshaped)

            return image, tf.pad(reshaped, [[0, tf.maximum(0, 4 * fc - tf.size(reshaped))]])

        return prepare_sample

    def _get_data_from_tf_record(self, file: str, cache_file, take=0):
        raw_dataset = tf.data.TFRecordDataset(file)
        parsed_dataset = raw_dataset.map(self.parse_tfrecord_fn()) \
            .filter(self.filter_amount_faces(self.faces_count)) \
            .cache(cache_file) \
            .shuffle(480) \
            .map(self._prepare_sample(self.faces_count)) \
            .prefetch(tf.data.AUTOTUNE)
        return parsed_dataset

    def get_dataset_splits(self):
        return self._get_data_from_tf_record(self.train_data_path, 'train.cache'), \
               self._get_data_from_tf_record(self.test_data_path, 'test.cache'), \
               self._get_data_from_tf_record(self.test_data_path, 'test.cache')
