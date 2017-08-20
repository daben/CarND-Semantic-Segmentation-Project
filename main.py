import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger("segmentation")
logger.setLevel(logging.INFO)


import os.path
import math
import time
import datetime
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    logger.warning('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

tf.set_random_seed(20170818)


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'

    tensor_names = ('image_input:0', 
                    'keep_prob:0', 
                    'layer3_out:0', 
                    'layer4_out:0', 
                    'layer7_out:0')

    model = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    return tuple(map(graph.get_tensor_by_name, tensor_names))

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    def conv1x1(inputs, num_outputs, name):
        return tf.layers.conv2d(inputs, num_outputs, kernel_size=1, strides=1,
                                name=name, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    def deconv2d(inputs, filters, kernel_size, strides, name):
        return tf.layers.conv2d_transpose(inputs, filters,
                    kernel_size=kernel_size, strides=(strides, strides), 
                    padding="same", name=name, 
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    # FCN-8 Decoder
    with tf.variable_scope("fcn_decoder"):
        # Replace the linear layers by 1x1 convolutions
        layer7_logits = conv1x1(vgg_layer7_out, num_classes, name="vgg_layer7_logits")
        layer4_logits = conv1x1(vgg_layer4_out, num_classes, name="vgg_layer4_logits")
        layer3_logits = conv1x1(vgg_layer3_out, num_classes, name="vgg_layer3_logits")
        # Upsample x2 the input to the original image size.
        layer1 = deconv2d(layer7_logits, num_classes, 4, 2, name="layer1")
        # skip connections
        layer2 = deconv2d(tf.add(layer1, layer4_logits), num_classes, 4, 2, name="layer2")
        layer3 = deconv2d(tf.add(layer2, layer3_logits), num_classes, 16, 8, name="layer3")

    # decoder output layer
    return layer3

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes, global_step=None):
    """
    Build the TensorFlow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    
    # Make logits and labels 2D
    # each row represents a pixel and each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    # Cross entropy loss
    cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    tf.summary.scalar("cross_entropy_loss", cross_entropy_loss)

    if False: # Try to maximize also IoU
        tf.add_to_collection("losses", cross_entropy_loss)
        
        # IoU loss
        iou, iou_op = tf.metrics.mean_iou(tf.argmax(labels, 1), tf.argmax(logits, 1), num_classes)
        with tf.control_dependencies([iou_op]):
            iou_loss = tf.subtract(tf.constant(1.0), iou)
        tf.add_to_collection("losses", iou_loss)
        tf.summary.scalar("mean_iou_loss", iou_loss)

        # Final loss function
        loss = tf.reduce_sum(tf.stack(tf.get_collection("losses")), name='loss')
    else:
        loss = cross_entropy_loss

    # Evaluation
    # correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    # tf.summary.scalar('accuracy', accuracy)

    # Our dataset is too small to fine tune the VGG layers. Here we select only the decoder
    # layers and we will pass this list to the optimizer.
    trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fcn_decoder/")
    if not len(trainable):
        # The scope is not defined, train everything. This will happen in the test function.
        trainable = None

    # Train operation
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=trainable, global_step=global_step)

    return logits, train_op, loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, **kwargs):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    # Keep compatibility with project_tests.py
    loss = cross_entropy_loss

    log_summary = 'log_summary' in kwargs

    if log_summary:
        logs_dir = kwargs['logs_dir']
        logits = kwargs['logits']
        image_shape = kwargs['image_shape']
        #testing_data_dir = kwargs['testing_data_dir']

        # Place holder for output image
        output_images = tf.placeholder(tf.uint8, (None, image_shape[0], image_shape[1], 3))
        tf.summary.image('output_images', output_images)

        output_labels = tf.placeholder(tf.uint8, (None, image_shape[0], image_shape[1], 3))
        tf.summary.image('output_labels', output_labels)

        # Merge all summaries
        summary_op = tf.summary.merge_all()
        train_log = tf.summary.FileWriter(logs_dir, sess.graph)


    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    time_elapsed_avg = 0.0
    step = 0
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_samples = 0
        time_start = time.perf_counter()

        for images, gt_images in get_batches_fn(batch_size):

            train_feed = {input_image: images,
                          correct_label: gt_images,
                          keep_prob: 0.75}

            _, batch_loss = sess.run([train_op, loss], feed_dict=train_feed)

            if log_summary:

                sample_images = helper.generate_street_images(sess, images[:3], logits, keep_prob, input_image, image_shape)
                sample_labels = tuple(helper.paste_segmentation(images[i], gt_images[i]) for i in range(3))

                summary_feed = train_feed.copy()
                summary_feed.update({output_images: sample_images,
                                     output_labels: sample_labels,
                                     keep_prob: 1.0})

                results = sess.run([summary_op], summary_feed)

                summary = results[0]
                train_log.add_summary(summary, step)  

            # Accumulated loss
            n_samples = len(images)
            epoch_loss += batch_loss * n_samples
            epoch_samples += n_samples
            step += 1

        epoch_loss /= epoch_samples

        # Logging
        time_end = time.perf_counter()
        time_elapsed = time_end - time_start
        time_elapsed_avg = (time_elapsed_avg * epoch + time_elapsed) / (epoch + 1)
        time_remaining = datetime.timedelta(seconds=math.ceil((epochs - epoch - 1) * time_elapsed_avg))

        logger.info("EPOCH {:3d}... Loss = {:6.3f}; {:.1f} sec ETA {}".format(
                    epoch + 1, epoch_loss, time_elapsed, time_remaining))


tests.test_train_nn(train_nn)



def run():
    # Parameters
    num_classes = 2
    image_shape = (160, 576)
    data_dir = 'data'
    runs_dir = 'runs'
    logs_dir = 'logs'

    tests.test_for_kitti_dataset(data_dir)
    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)
    
    training_data_dir = os.path.join(data_dir, 'data_road', 'training')
    testing_data_dir = os.path.join(data_dir, 'data_road', 'testing')

    print("------------------------------------------", flush=True)

    # FLAGS
    epochs = 100
    batch_size = 16
    num_augmented_batches = 100

    # training data size
    train_size = sum(1 for _ in os.scandir(os.path.join(training_data_dir, 'image_2'))
                        if _.is_file() and _.name.endswith('.png'))
    # Total number of batches across all epochs
    #total_steps = math.ceil(train_size / batch_size) * epochs
    total_steps = num_augmented_batches * epochs

    logger.info("Training...")

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(training_data_dir, image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        get_batches_fn = helper.gen_augmented_batch_function(get_batches_fn, num_augmented_batches)

        # Build NN using load_vgg, layers, and optimize function

        # Load VGG
        image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        # Build FCN-8
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        
        # Build the training operation
        
        # Placeholder for the labels
        correct_label = tf.placeholder(tf.float32, shape=(None, image_shape[0], image_shape[1], num_classes), name='correct_label')

        # Step counter during training
        global_step = tf.Variable(0, trainable=False, name='step')

        # Decay learning rate, once per step, with an exponential schedule.
        learning_rate = tf.train.exponential_decay(
            learning_rate=1e-3, 
            global_step=global_step,
            decay_steps=total_steps, 
            decay_rate=0.3,  # final learning rate = learning * decay_rate 
            name="learning_rate")
        tf.summary.scalar("learning_rate", learning_rate)

        # Other option is to keep it constant
        #learning_rate = tf.constant(5e-4)

        logits, train_op, loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes, global_step)

        # Train
        extra_kwargs = dict(
            log_summary=True,
            logs_dir=logs_dir, 
            logits=logits, 
            image_shape=image_shape,
            testing_data_dir=testing_data_dir)

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, loss,
                 image_input, correct_label, keep_prob, learning_rate, **extra_kwargs)

        # Save inference data using helper.save_inference_samples
        logger.info("Saving inference samples and checkpoint")
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)


if __name__ == '__main__':
    run()