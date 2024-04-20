#================================================================
#
#   File name   : yolov3.py
#   Author      : PyLessons
#   Created date: 2020-06-04
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : main yolov3 functions
#
#================================================================
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Input, LeakyReLU, ZeroPadding1D, BatchNormalization, MaxPool1D
from tensorflow.keras.regularizers import l2

NUM_CLASS = 3
TRAIN_YOLO_TINY = True

STRIDES = [8, 16, 32]
ANCHORS = [
    [10, 50, 100],
    [70, 110, 200],
    [150, 300, 450]
]

IOU_LOSS_THRESH = 0.5



class BatchNormalization(BatchNormalization):
    # "Frozen state" and "inference mode" are two separate concepts.
    # `layer.trainable = False` is to freeze the layer, so the layer will use
    # stored moving `var` and `mean` in the "inference mode", and both `gama`
    # and `beta` will not be updated !
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True):
    if downsample:
        input_layer = ZeroPadding1D((1, 0))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = Conv1D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides,
                  padding=padding, use_bias=not bn, kernel_regularizer=l2(0.0005),
                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                  bias_initializer=tf.constant_initializer(0.))(input_layer)
    if bn:
        conv = BatchNormalization()(conv)
    if activate == True:
        conv = LeakyReLU(alpha=0.1)(conv)

    return conv

def residual_block(input_layer, input_channel, filter_num1, filter_num2):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, input_channel, filter_num1))
    conv = convolutional(conv       , filters_shape=(3, filter_num1,   filter_num2))

    residual_output = short_cut + conv
    return residual_output

def upsample(input_layer):
    return tf.image.resize(input_layer, input_layer.shape[1] * 2, method='nearest')


def darknet53(input_data):
    input_data = convolutional(input_data, (3,  3,  32))
    input_data = convolutional(input_data, (3, 32,  64), downsample=True)

    for i in range(1):
        input_data = residual_block(input_data,  64,  32, 64)

    input_data = convolutional(input_data, (3,  64, 128), downsample=True)

    for i in range(2):
        input_data = residual_block(input_data, 128,  64, 128)

    input_data = convolutional(input_data, (3, 128, 256), downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 256, 128, 256)

    route_1 = input_data
    input_data = convolutional(input_data, (3, 256, 512), downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 512, 256, 512)

    route_2 = input_data
    input_data = convolutional(input_data, (3, 512, 1024), downsample=True)

    for i in range(4):
        input_data = residual_block(input_data, 1024, 512, 1024)

    return route_1, route_2, input_data

def darknet19_tiny(input_data):
    input_data = convolutional(input_data, (3, 3, 16))
    input_data = MaxPool1D(2, 2, 'same')(input_data)
    input_data = convolutional(input_data, (3, 16, 32))
    input_data = MaxPool1D(2, 2, 'same')(input_data)
    input_data = convolutional(input_data, (3, 32, 64))
    input_data = MaxPool1D(2, 2, 'same')(input_data)
    input_data = convolutional(input_data, (3, 64, 128))
    input_data = MaxPool1D(2, 2, 'same')(input_data)
    input_data = convolutional(input_data, (3, 128, 256))
    route_1 = input_data
    input_data = MaxPool1D(2, 2, 'same')(input_data)
    input_data = convolutional(input_data, (3, 256, 512))
    input_data = MaxPool1D(2, 1, 'same')(input_data)
    input_data = convolutional(input_data, (3, 512, 1024))

    return route_1, input_data

def YOLOv3(input_layer, NUM_CLASS):
    # After the input layer enters the Darknet-53 network, we get three branches
    route_1, route_2, conv = darknet53(input_layer)
    # See the orange module (DBL) in the figure above, a total of 5 Subconvolution operation
    conv = convolutional(conv, (1, 1024,  512))
    conv = convolutional(conv, (3,  512, 1024))
    conv = convolutional(conv, (1, 1024,  512))
    conv = convolutional(conv, (3,  512, 1024))
    conv = convolutional(conv, (1, 1024,  512))
    conv_lobj_branch = convolutional(conv, (3, 3, 512, 1024))
    
    # conv_lbbox is used to predict large-sized objects , Shape = [None, 13, 13, 255] 
    conv_lbbox = convolutional(conv_lobj_branch, (1, 1, 1024, 3*(NUM_CLASS + 5)), activate=False, bn=False)

    conv = convolutional(conv, (1,  512,  256))
    # upsample here uses the nearest neighbor interpolation method, which has the advantage that the
    # upsampling process does not need to learn, thereby reducing the network parameter  
    conv = upsample(conv)

    conv = tf.concat([conv, route_2], axis=-1)
    conv = convolutional(conv, (1, 768, 256))
    conv = convolutional(conv, (3, 256, 512))
    conv = convolutional(conv, (1, 512, 256))
    conv = convolutional(conv, (3, 256, 512))
    conv = convolutional(conv, (1, 512, 256))
    conv_mobj_branch = convolutional(conv, (3, 3, 256, 512))

    # conv_mbbox is used to predict medium-sized objects, shape = [None, 26, 26, 255]
    conv_mbbox = convolutional(conv_mobj_branch, (1, 512, 3*(NUM_CLASS + 5)), activate=False, bn=False)

    conv = convolutional(conv, (1, 256, 128))
    conv = upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)
    conv = convolutional(conv, (1, 384, 128))
    conv = convolutional(conv, (3, 128, 256))
    conv = convolutional(conv, (1, 256, 128))
    conv = convolutional(conv, (3, 128, 256))
    conv = convolutional(conv, (1, 256, 128))
    conv_sobj_branch = convolutional(conv, (3, 128, 256))
    
    # conv_sbbox is used to predict small size objects, shape = [None, 52, 52, 255]
    conv_sbbox = convolutional(conv_sobj_branch, ( 1, 256, 3*(NUM_CLASS + 3)), activate=False, bn=False)
        
    return [conv_sbbox, conv_mbbox, conv_lbbox]

def YOLOv3_tiny(input_layer, NUM_CLASS):
    # After the input layer enters the Darknet-53 network, we get three branches
    route_1, conv = darknet19_tiny(input_layer)

    conv = convolutional(conv, (1, 1024, 256))
    conv_lobj_branch = convolutional(conv, (3, 256, 512))
    
    # conv_lbbox is used to predict large-sized objects , Shape = [None, 26, 26, 255]
    conv_lbbox = convolutional(conv_lobj_branch, (1, 512, 3*(NUM_CLASS + 3)), activate=False, bn=False)

    conv = convolutional(conv, (1, 256, 128))
    # upsample here uses the nearest neighbor interpolation method, which has the advantage that the
    # upsampling process does not need to learn, thereby reducing the network parameter  
    conv = upsample(conv)
    
    conv = tf.concat([conv, route_1], axis=-1)
    conv_mobj_branch = convolutional(conv, (3, 128, 256))
    # conv_mbbox is used to predict medium size objects, shape = [None, 13, 13, 255]
    conv_mbbox = convolutional(conv_mobj_branch, (1, 256, 3 * (NUM_CLASS + 3)), activate=False, bn=False)

    return [conv_mbbox, conv_lbbox]

def Create_Yolov3(input_size=416, channels=3, training=False):
    input_layer  = Input([input_size, input_size, channels])

    if TRAIN_YOLO_TINY:
        conv_tensors = YOLOv3_tiny(input_layer, NUM_CLASS)
    else:
        conv_tensors = YOLOv3(input_layer, NUM_CLASS)

    output_tensors = []
    for i, conv_tensor in enumerate(conv_tensors):
        pred_tensor = decode(conv_tensor, NUM_CLASS, i)
        if training: output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)

    YoloV3 = tf.keras.Model(input_layer, output_tensors)
    return YoloV3

def decode(conv_output, NUM_CLASS, i=0):
    # where i = 0, 1 or 2 to correspond to the three grid scales  
    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]

    conv_output = tf.reshape(conv_output, (batch_size, output_size, 3, 3 + NUM_CLASS))

    conv_raw_dx   = conv_output[:, :, :, 0] # offset of center position     
    conv_raw_dw   = conv_output[:, :, :, 1] # Prediction box length and width offset
    conv_raw_conf = conv_output[:, :, :, 2] # confidence of the prediction box
    conv_raw_prob = conv_output[:, :, :, 3:] # category probability of the prediction box 

    # next need Draw the grid. Where output_size is equal to 13, 26 or 52  
    x_grid = tf.range(output_size,dtype=tf.int32)
    x_grid = tf.tile(x_grid[tf.newaxis, :, tf.newaxis, :], [batch_size, 1, 3, 1])
    x_grid = tf.cast(x_grid, tf.float32)

    # Calculate the center position of the prediction box:
    pred_x = (tf.sigmoid(conv_raw_dx) + x_grid) * STRIDES[i]
    # Calculate the length and width of the prediction box:
    pred_w = (tf.exp(conv_raw_dw) * ANCHORS[i]) * STRIDES[i]

    pred_xw = tf.concat([pred_x, pred_w], axis=-1)
    pred_conf = tf.sigmoid(conv_raw_conf) # object box calculates the predicted confidence
    pred_prob = tf.sigmoid(conv_raw_prob) # calculating the predicted probability category box object

    # calculating the predicted probability category box object
    return tf.concat([pred_xw, pred_conf, pred_prob], axis=-1)

def bbox_iou(boxes1, boxes2):
    boxes1_width = boxes1[..., 1]
    boxes2_width = boxes2[..., 1]

    boxes1 = tf.concat([boxes1[..., :1] - boxes1[..., 1:] * 0.5,
                        boxes1[..., :1] + boxes1[..., 1:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :1] - boxes2[..., 1:] * 0.5,
                        boxes2[..., :1] + boxes2[..., 1:] * 0.5], axis=-1)

    left = tf.maximum(boxes1[..., :1], boxes2[..., :1])
    right = tf.minimum(boxes1[..., 1:], boxes2[..., 1:])

    inter_section = tf.maximum(right - left, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_width + boxes2_width - inter_area

    return 1.0 * inter_area / union_area

def bbox_giou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :1] - boxes1[..., 1:] * 0.5,
                        boxes1[..., :1] + boxes1[..., 1:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :1] - boxes2[..., 1:] * 0.5,
                        boxes2[..., :1] + boxes2[..., 1:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :1], boxes1[..., 1:]),
                        tf.maximum(boxes1[..., :1], boxes1[..., 1:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :1], boxes2[..., 1:]),
                        tf.maximum(boxes2[..., :1], boxes2[..., 1:])], axis=-1)

    boxes1_width = (boxes1[..., 1] - boxes1[..., 0])
    boxes2_width = (boxes2[..., 1] - boxes2[..., 0])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_width + boxes2_width - inter_area

    # Calculate the iou value between the two bounding boxes
    iou = inter_area / union_area

    # Calculate the coordinates of the upper left corner and the lower right corner of the smallest closed convex surface
    enclose_left = tf.minimum(boxes1[..., :1], boxes2[..., :1])
    enclose_right = tf.maximum(boxes1[..., 1:], boxes2[..., 1:])
    enclose = tf.maximum(enclose_right - enclose_left, 0.0)

    # Calculate the area of the smallest closed convex surface C
    enclose_area = enclose[..., 0]

    # Calculate the GIoU value according to the GioU formula  
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


def compute_loss(pred, conv, label, bboxes, i=0):
    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, 3, 3 + NUM_CLASS))

    conv_raw_conf = conv[:, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, 5:]

    pred_xw       = pred[:, :, :, 0:4]
    pred_conf     = pred[:, :, :, 4:5]

    label_xw      = label[:, :, :, 0:4]
    respond_bbox  = label[:, :, :, 4:5]
    label_prob    = label[:, :, :, 5:]

    giou = tf.expand_dims(bbox_giou(pred_xw, label_xw), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xw[:, :, :, 1:2] / (input_size)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    iou = bbox_iou(pred_xw[:, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :])
    # Find the value of IoU with the real box The largest prediction box
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    # If the largest iou is less than the threshold, it is considered that the prediction box contains no objects, then the background box
    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < IOU_LOSS_THRESH, tf.float32 )

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    # Calculate the loss of confidence
    # we hope that if the grid contains objects, then the network output prediction box has a confidence of 1 and 0 when there is no object.
    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    return giou_loss, conf_loss, prob_loss