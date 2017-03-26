import os
import os.path
import numpy as np
import glob
import tensorflow as tf
import cv2

#generate_records:
def extract_image(filename, resize, resize_height, resize_width):
  image = cv2.imread(filename)
	if resize:
		image = cv2.resize(image, (resize_height, resize_width))
	# transform bgr to rgb
	b, g, r = cv2.split(image)  # get b,g,r
	rgb_image = cv2.merge([r, g, b])  # switch it to rgb
	return rgb_image

writer = tf.python_io.TFRecordWriter('train.tfrecords')
for img_path in enumerate(training_list):  # file path list
  image = extract_image(path_path, True, 224, 224)   # 224 is for resize_height and rezise_width. you can change for your need.
  image_raw = image.tostring()
  example = tf.train.Example(features=tf.train.Features(feature={
			'image_raw': _bytes_features(image_raw),
			'label': _int64_features(training_label[index]),
			'height': _int64_features(image.shape[0]),
			'width': _int64_features(image.shape[1]),
			'channels': _int64_features(image.shape[2])
		  }))
  writer.write(example.SerializeToString())
  print("...file: %s" % (path))
writer.close()

#read_records:
files = tf.train.match_filenames_once('./data/flower_photos/train.tfrecords')
filename_queue = tf.train.string_input_producer(files, shuffle=False)
reader = tf.TFRecordReader()
_,serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw':tf.FixedLenFeature([], tf.string),
                'label':tf.FixedLenFeature([], tf.int64)
                })
decoded_images = tf.decode_raw(features['image_raw'], tf.uint8)
print(decoded_images.shape)
retyped_images = tf.cast(decoded_images, tf.float32)
labels = tf.cast(features['label'], tf.int32)
images = tf.reshape(retyped_images, [224 * 224 * 3])

print(retyped_images.shape)
print(type(images))
print(images.shape)
