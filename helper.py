import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import matplotlib.colors
import scipy.ndimage


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def paste_segmentation(image, segmentation):
    if segmentation.shape[-1] == 2:
        segmentation = segmentation[..., 1:2]
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]], dtype=np.uint8))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)
    return scipy.misc.fromimage(street_im, mode="RGB")

def generate_street_images(sess, images, logits, keep_prob, input_image, image_shape):
    results = sess.run([tf.nn.softmax(logits)], {input_image: images, keep_prob: 1.0})
    im_softmax = results[0]
    im_softmax = im_softmax[:, 1].reshape(len(images), image_shape[0], image_shape[1])
    street_images = np.zeros((len(images), image_shape[0], image_shape[1], 3), dtype=np.uint8)
    for i in range(len(images)):
        segmentation = (im_softmax[i] > 0.5).reshape(image_shape[0], image_shape[1], 1)
        street_images[i] = paste_segmentation(images[i], segmentation)
    return street_images

def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

    # Save model checkpoint
    saver = tf.train.Saver()
    save_path = saver.save(sess, os.path.join(output_dir, 'fcn.ckpt'))
    print('Model checkpoint saved to: {}'.format(save_path))


def augment_shadow(image):
    h, w = image.shape[0], image.shape[1]
    x1, x2 = np.random.choice(w, 2, replace=False)
    k = h / (x2 - x1)
    b = - k * x1
    random_bright = random.uniform(.4, .6)
    shadowed = image.copy()
    for i in range(h):
        c = int((i - b) / k)
        shadowed[i, :c, :] = np.clip(image[i, :c, :] * random_bright, 0, 255)
    return shadowed

def augment_brightness(image, bright=None):
    if bright is None:
        bright = random.uniform(.25, 1.5)
    hsv = matplotlib.colors.rgb_to_hsv(image/255)
    hsv[..., 2] = hsv[..., 2] * bright
    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)

def augment_scale(image, mask, mode='constant'):
    s = random.uniform(0.8, 1)
    matrix = np.eye(2,2) * s
    offset = 0.5 * np.array(image.shape[:2])
    offset -= np.dot(offset, matrix)
    
    transformed_image = np.zeros_like(image)
    for i in range(image.shape[-1]):
        transformed_image[..., i] = scipy.ndimage.affine_transform(image[..., i], matrix, offset, mode=mode)

    transformed_mask = np.zeros_like(mask)
    for i in range(mask.shape[2]):
        transformed_mask[..., i] = scipy.ndimage.affine_transform(mask[..., i] * 255, matrix, offset, mode=mode) != 0

    return transformed_image, transformed_mask

def augment_image(image, label):
    image = augment_brightness(image)
    image = augment_shadow(image)
    # FIXME This is not working
    # image, label = augment_scale(image, label)
    if random.random() < 0.5:
        # Flip horizontally
        image = image[:, ::-1, :]
        label = label[:, ::-1, :]
    return image, label

def augment_images(images, labels):
    target_images = np.zeros_like(images)
    target_labels = np.zeros_like(labels)
    for i in range(len(images)):
        target_images[i], target_labels[i] = augment_image(images[i], labels[i])
    return target_images, target_labels


def gen_augmented_batch_function(get_batches_fn, num_batches):
    def get_augmented_batches_fn(batch_size):
        for batch in range(num_batches):
            images, gt_images = next(get_batches_fn(batch_size))
            aug_images, aug_gt_images = augment_images(images, gt_images)
            yield aug_images, aug_gt_images
    return get_augmented_batches_fn