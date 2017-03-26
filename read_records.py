import tensorflow as tf
import os
gpuconfig = tf.ConfigProto(
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5),
        )

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

with tf.Graph().as_default(), tf.Session(config=gpuconfig) as sess:
    #files = tf.train.match_filenames_once('../practice/mnist/data/output.tfrecords')
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
    min_after_dequeue = 10
    batch_size = 2
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch([images, labels],
            batch_size=batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue)
    def inference(input_tensor, weights1, biases1, weights2, biases2):
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    INPUT_NODE = 224 * 224 * 3
    OUTPUT_NODE = 10
    LAYER1_NODE = 500
    REGULARAZTION_RATE = 0.0001
    TRAINING_STEPS = 50

    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    y = inference(image_batch, weights1, biases1, weights2, biases2)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels = label_batch)
    cross_entropy_mean =  tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    regularazation = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularazation

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    tf.global_variables_initializer().run()
    print('...training start')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(TRAINING_STEPS):
        if i % 10 == 0:
            print('After %d training step(s), loss is %g' % (i, sess.run(loss)))
        sess.run(train_step)
    coord.request_stop()
    coord.join(threads)
