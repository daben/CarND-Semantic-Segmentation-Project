import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger("segmentation")

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




def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'

    vgg_model = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()

    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    vgg_input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob_tensor = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out_tensor = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out_tensor = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out_tensor = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor

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

    # FCN-8 Decoder

    # Replace the linear layers by 1x1 convolutions
    layer7_logits = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=1, strides=(1,1))
    layer4_logits = tf.layers.conv2d(vgg_layer4_out, num_classes, kernel_size=1, strides=(1,1))   
    layer3_logits = tf.layers.conv2d(vgg_layer3_out, num_classes, kernel_size=1, strides=(1,1))

    # Upsample the input to the original image size.
    output = tf.layers.conv2d_transpose(layer7_logits, num_classes, 
                                        kernel_size=4, strides=(2, 2), padding='same')

    # skip connections

    # fuse layers
    output = tf.add(output, layer4_logits)
    output = tf.layers.conv2d_transpose(output, num_classes, 
                                        kernel_size=4, strides=(2, 2), padding='same')

    # fuse layers
    output = tf.add(output, layer3_logits)
    output = tf.layers.conv2d_transpose(output, num_classes, 
                                        kernel_size=16, strides=(8, 8), padding='same')

    # decoder output layer
    return output

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
    
    # Make logits 2D
    # each row represents a pixel and each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    # Retrieve the labels vector
    labels = tf.argmax(tf.reshape(correct_label, (-1, num_classes)), 1)

    # Cross entropy loss
    # cross_entropy_loss = tf.reduce_mean(
    #     tf.nn.sparse_softmax_cross_entropy_with_logits(labels=correct_label, logits=logits))
    cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits=logits))

    tf.add_to_collection("losses", cross_entropy_loss)
    tf.summary.scalar('cross_entropy', cross_entropy_loss)

    # IoU loss
    iou, iou_op = tf.metrics.mean_iou(labels, tf.argmax(logits, 1), num_classes)
    with tf.control_dependencies([iou_op]):
        iou_loss = tf.subtract(tf.constant(1.0), iou)
        tf.add_to_collection("losses", iou_loss)
    tf.summary.scalar('mean_iou', iou)

    # Compute final loss function
    loss = tf.add_n(tf.get_collection("losses"), name='loss')

    # Evaluation
    # correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    # tf.summary.scalar('accuracy', accuracy)

    # Train operation
    train_op = (tf.train.AdamOptimizer(learning_rate).
                    minimize(loss, global_step=global_step))

    return logits, train_op, loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss,
             input_image, correct_label, keep_prob, learning_rate, logs_dir=None, total_steps=0):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    :param logs_dir: path where train summary is written
    :param total_steps: number of total steps
    """

    log_summary = False

    if logs_dir is not None:
        # Merge all summaries
        summary_op = tf.summary.merge_all()        
    
        if summary_op is not None:
            train_log = tf.summary.FileWriter(logs_dir, sess.graph)
            log_summary = True
    
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    # Required by mean_iou
    sess.run(tf.local_variables_initializer())


    loss = cross_entropy_loss
    time_elapsed_avg = 0.0

    step = 0
    for epoch in range(epochs):
        
        for images, gt_images in get_batches_fn(batch_size):            
            time_start = time.perf_counter()

            train_feed = {input_image: images,
                          correct_label: gt_images,
                          keep_prob: 0.8}

            if log_summary:

                step_loss, summary, _ = sess.run([loss, summary_op, train_op], feed_dict=train_feed)
                train_log.add_summary(summary, step)

            else:
                step_loss, _ = sess.run([loss, train_op], feed_dict=train_feed)
    
            if logs_dir and step % 10 == 0:
                # TODO: Evaluate accuracy on test data
                # summary, _ = sess.run([summary_op, accuracy], feed_dict=test_feed)
                # test_writer.add_summary(summary, step)

                time_end = time.perf_counter()
                time_elapsed = time_end - time_start
                time_elapsed_avg = (time_elapsed_avg * step + time_elapsed) / (step + 1)
                time_remaining = datetime.timedelta(seconds=math.ceil((total_steps - step - 1) * time_elapsed_avg))

                logger.info("EPOCH {:3d}...  Step = {:4d}  Loss = {:7.3f}; {:.1f} sec ETA {}".format(
                    epoch, step, step_loss, time_elapsed, time_remaining))

            step += 1

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

    logger.info("Training...")

    # Flags
    epochs = 100
    batch_size = 16

    learning_rate_start = 1e-3
    learning_rate_decay = 0.1
    
    training_data_dir = os.path.join(data_dir, 'data_road', 'training')

    # Step counter during training
    global_step = tf.Variable(0, trainable=False, name='step')

    if learning_rate_decay:
        # Total number of batches across all epochs
        train_size = sum(1 for _ in os.scandir(os.path.join(training_data_dir, 'image_2'))
                            if _.is_file() and _.name.endswith('.png'))
        total_steps = math.ceil(train_size / batch_size) * epochs

        # Decay learning rate, once per step, with an exponential schedule.
        learning_rate = tf.train.exponential_decay(
           learning_rate_start, 
           global_step,
           decay_steps=total_steps, 
           decay_rate=learning_rate_decay, 
           staircase=False, 
           name='learning_rate')

        tf.summary.scalar('learning_rate', learning_rate)
    else:
        # Constant learning rate
        learning_rate = tf.constant(learning_rate_start, dtype=tf.float32)



    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    # TensorFlow configuration object.
    config = tf.ConfigProto()
    # JIT level, this can be set to ON_1 or ON_2 
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    with tf.Session(config=config) as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(training_data_dir, image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize functions
        correct_label = tf.placeholder(tf.float32, shape=(None,)+image_shape+(num_classes,), name="correct_label")

        # Load VGG
        image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        # Build FCN-8
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        # Build the training operation
        logits, train_op, loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes, global_step)

        # Train
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, loss,
                 image_input, correct_label, keep_prob, learning_rate, 
                 logs_dir, total_steps)
    
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, 
                                      keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
    