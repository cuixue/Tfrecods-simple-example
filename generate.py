import os
import os.path
import numpy as np
import glob
import tensorflow as tf
from PIL import Image
import cv2

INPUT_DATA = './data/flower_photos/roses'

TESTING_PERCENTAGE = 10
VALIDATION_PERCENTAGE = 10

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

def create_image_lists(testing_percentage, validation_percentage):
	images_lists = {}
	label_names = []
	sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
	is_root_dir = False
	for sub_dir in sub_dirs:
		if is_root_dir:
			is_root_dir = False
			continue
		extensions=['jpg','jpeg','JPG','JPEG']
		file_list = []
		dir_name = os.path.basename(sub_dir)
		for extension in extensions:
			file_glob = os.path.join(INPUT_DATA,  '*.' + extension)
			file_list.extend(glob.glob(file_glob))
		#if not file_list: continue
		label_name = dir_name.lower()
		label_names.append(label_name)
		training_images = []
		testing_images = []
		validation_images = []
		for file_name in file_list:
			base_name = os.path.basename(file_name)
			chance = np.random.randint(100)
			if chance < validation_percentage:
				validation_images.append(base_name)
			elif chance < (testing_percentage + validation_percentage):
				testing_images.append(base_name)
			else:
				training_images.append(base_name)
		images_lists[label_name] = {
			'dir':dir_name,
			'training':training_images,
			'testing':testing_images,
			'validation':validation_images,
		}
	return images_lists,label_names


def image2tfrecord(sess, image_lists, label_names):
	training_list = []
	training_label = []
	testing_list = []
	testing_label = []
	validation_list = []
	validation_label = []

	for i,label_name in enumerate(label_names):
		image_list = image_lists[label_name]
		# for training image
		training_lists = image_list['training']
		for training_image in training_lists:
			training_list.append(training_image)
			training_label.append(i)
	# for testing image
		testing_lists = image_list['testing']
		for testing_image in testing_lists:
			testing_list.append(testing_image)
			testing_label.append(i)
		# for training image
		validation_lists = image_list['validation']
		for validation_image in validation_lists:
			validation_list.append(validation_image)
			validation_label.append(i)

	def _int64_features(value):
		return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
	def _bytes_features(value):
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
	def preprocess_for_image(image, height, width):
		if image.dtype != tf.float32:
			image = tf.image.convert_image_dtype(image, dtype=tf.float32)
		#distorted_image = tf.image.resize_images(image, [height, width,], method=0)
		return image

	def extract_image(filename, resize, resize_height, resize_width):
		image = cv2.imread(filename)
		if resize:
			image = cv2.resize(image, (resize_height, resize_width))

		# transform bgr to rgb
		b, g, r = cv2.split(image)  # get b,g,r
		rgb_image = cv2.merge([r, g, b])  # switch it to rgb

		return rgb_image

	## store training set
	training_filename = os.path.join(INPUT_DATA, 'train6.tfrecords')
	writer = tf.python_io.TFRecordWriter(training_filename)
	for index,img in enumerate(training_list):
		path = os.path.join(INPUT_DATA,training_list[index])
		image = extract_image(path, True, 224, 224)
		image_raw = image.tostring()
		example = tf.train.Example(features=tf.train.Features(feature={
			'image_raw': _bytes_features(image_raw),
			'label': _int64_features(training_label[index]),
			'pixels': _int64_features(240*320*3),
		}))
		writer.write(example.SerializeToString())
		print("...file: %s" % (path))
	writer.close()
	print('...training set done.')
	## store testing set
	testing_filename = os.path.join(INPUT_DATA, 'test.tfrecords')
	writer = tf.python_io.TFRecordWriter(testing_filename)
	for index,img in enumerate(testing_list):
		path = os.path.join(INPUT_DATA, label_names[testing_label[index]],testing_list[index])
		image = tf.gfile.FastGFile(path, 'r').read()
		img_data = tf.image.decode_jpeg(image)
		image_raw_data = preprocess_for_image(img_data, IMAGE_HEIGHT, IMAGE_WIDTH)
		img_raw = sess.run(image_raw_data)

		image_raw = img_raw.tostring()
		example = tf.train.Example(features=tf.train.Features(feature={
			'image_raw': _bytes_features(image_raw),
			'label': _int64_features(testing_label[index]),
			'height': _int64_features(img_raw.shape[0]),
			'width': _int64_features(img_raw.shape[1]),
			'channels': _int64_features(img_raw.shape[2])
		}))
		writer.write(example.SerializeToString())
		print("...file: %s" % (path))
	writer.close()
	print('...testing set done.')
	## store validation set
	validation_filename = os.path.join(INPUT_DATA, 'validation.tfrecords')
	writer = tf.python_io.TFRecordWriter(validation_filename)
	for index,img in enumerate(validation_list):
		path = os.path.join(INPUT_DATA, label_names[validation_label[index]],validation_list[index])
		image = tf.gfile.FastGFile(path, 'r').read()
		img_data = tf.image.decode_jpeg(image)
		image_raw_data = preprocess_for_image(img_data, IMAGE_HEIGHT, IMAGE_WIDTH)
		#img_raw = sess.run(image_raw_data)

		image_raw = image_raw_data.tostring()
		example = tf.train.Example(features=tf.train.Features(feature={
			'image_raw': _bytes_features(image_raw),
			'label': _int64_features(validation_label[index]),
			'height': _int64_features(img_raw.shape[0]),
			'width': _int64_features(img_raw.shape[1]),
			'channels': _int64_features(img_raw.shape[2])
		}))
		writer.write(example.SerializeToString())
		print("...file: %s" % (path))
	writer.close()
	print('...validation set done.')
	return 0
if __name__=='__main__':
	gpuconfig = tf.ConfigProto(
		gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5),
	)
	os.environ['CUDA_VISIBLE_DEVICES'] = '1'

	with tf.Graph().as_default(), tf.Session(config=gpuconfig) as sess:
		image_lists, label_names = create_image_lists(TESTING_PERCENTAGE, VALIDATION_PERCENTAGE)
		image2tfrecord(sess, image_lists,label_names)

