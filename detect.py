import os
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf #tensorflow_version 1.x
import time

#PARAMS
WIDTH_SIZE = 512
HEIGHT_SIZE = 384
MODEL_NAME ="model.pb"

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  #INPUT_SIZE = 512


  def __init__(self):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()
    graph_def = None
    with tf.io.gfile.GFile(os.path.join('./models/', MODEL_NAME), 'rb') as f:
        graph_def = tf.GraphDef.FromString(f.read())
        
    if graph_def is None:
      raise RuntimeError('Cannot find inference graph.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    #width, height = image.size
    #resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    width=WIDTH_SIZE
    height=HEIGHT_SIZE
    resize_ratio = 1.0 
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_pnoa_label_colormap():
  """Creates a label colormap

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [0, 0, 0]
  colormap[1] = [255, 255, 255]
  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pnoa_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def vis_segmentation(image, seg_map, image_name):
  """Visualizes input image, segmentation map and overlay view."""
  #plt.figure(figsize=(15, 5))
  #grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])
#
  #plt.subplot(grid_spec[0])
  #plt.imshow(image)
  #plt.axis('off')
  #plt.title('input image')
#
  #plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  #plt.imshow(seg_image)
  #plt.axis('off')
  #plt.title('segmentation map')
#
  #plt.subplot(grid_spec[2])
  #plt.imshow(image)
  #plt.imshow(seg_image, alpha=0.7)
  #plt.axis('off')
  #plt.title('segmentation overlay')
#
  #unique_labels = np.unique(seg_map)
  #ax = plt.subplot(grid_spec[3])
  #plt.imshow(
  #    FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  #ax.yaxis.tick_right()
  #plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  #plt.xticks([], [])
  #ax.tick_params(width=0.0)
  #plt.grid('off')
  ## Save resume with overlay
  #plt.savefig(image_name.replace('/images/','/detections/resume_'))
  #plt.close()
 ## Save raw segmentation image 
  seg_image_file = Image.fromarray(seg_image)
  seg_image_file.save(image_name.replace('/images/','/detections/seg_').replace('.jpg','').replace('png','') + '.png', format='png')


#LABEL_NAMES = np.asarray([
#    'BKG', 'LE'])

#FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
#FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

# Load pretrained 
MODEL = DeepLabModel()
print('model loaded successfully!')


def run_visualization(path):
  """Inferences DeepLab model and visualizes result."""
  try:
    original_im = Image.open(path)
  except IOError:
    print('Cannot retrieve image. Please check path: ' + path)
    return

  print('running deeplab on image %s...' % path)
  start = time.time()
  resized_im, seg_map = MODEL.run(original_im)
  end = time.time()
  print("Elapsed time: ",end - start)

  vis_segmentation(resized_im, seg_map, path)

# Run on images folder
for root, dirs, files in os.walk('./images', topdown=False):
  for name in files:
    image_path = './images/' + name
    run_visualization(image_path)
