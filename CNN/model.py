import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from sklearn.model_selection import train_test_split

NUM_INPUTS = 1000
NUM_CHANNELS = 1

NUM_CLASSES = 2
NUM_DIVS = 1

BATCH_SIZE = 1
NUM_EPOCHS = 5

test_input = tf.random.uniform((BATCH_SIZE, NUM_INPUTS, NUM_CHANNELS), dtype=float)
print ("Input size: ", tf.shape(test_input))

# Define the model architecture

'''
class Decode(tf.keras.Layer):
    def call(self, x):
        pred = tf.reshape(x, (BATCH_SIZE, NUM_DIVS, (NUM_CLASSES + 3)))
        pred_x     = tf.sigmoid(pred[:, :, 0:1])
        pred_w     = pred[:, :, 1:2]
        pred_conf  = tf.sigmoid(pred[:, :, 2:3])
        pred_class = tf.nn.softmax(pred[:, :, 3:], axis=-1)

        return tf.concat([pred_x, pred_w, pred_conf, pred_class], axis=-1)
'''

model = tf.keras.Sequential([

    Input((NUM_INPUTS, NUM_CHANNELS)),

    Conv1D(filters=32, kernel_size=3, activation='relu'),

    MaxPooling1D(pool_size=2),

    Conv1D(filters=64, kernel_size=3, activation='relu'),

    MaxPooling1D(pool_size=2),

    Flatten(),

    Dense(128, activation='relu'),

    Dropout(0.5),

    Dense(32, activation='relu'),

    Dropout(0.5),

    Dense(NUM_CLASSES*NUM_DIVS)

    #,Decode()

])


def bbox_iou(label, pred):
    width1 = label[:, :, :, 1]
    width2 = pred[:, :, :, 1]
    
    x_label = label[:, :, :, 0:1]
    w_label = label[:, :, :, 1:2]
    
    x_pred = pred[:, :, :, 0:1]
    w_pred = pred[:, :, :, 1:2]

    boxes1 = tf.concat([x_label - w_label * 0.5,
                        x_label + w_label * 0.5], axis=-1)
    boxes2 = tf.concat([x_pred - w_pred * 0.5,
                        x_pred + w_pred * 0.5], axis=-1)

    left = tf.maximum(boxes1[..., :1], boxes2[..., :1], axis=-1)
    right = tf.minimum(boxes1[..., 1:], boxes2[..., 1:], axis=-1)
    
    inter_section = tf.maximum(right - left, 0.0)
    inter_area = inter_section[..., 0]
    union_area = width1 + width2 - inter_area

    return 1.0 * inter_area / union_area
    




def compute_loss(label, pred):

    pred_conf = pred[..., 2:3]
    pred_prob = pred[..., 3:]

    label_conf = label[..., 2:3]
    label_prob = label[..., 3:]

    iou = bbox_iou(label, pred)
    iou_loss = label_conf * (1.0 - iou)

    conf_focal = tf.pow(label_conf - pred_conf, 2)

    # Calculate the loss of confidence
    # we hope that if the grid contains objects, then the network output prediction box has a confidence of 1 and 0 when there is no object.
    conf_loss = conf_focal * label_conf * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_conf, logits=pred_conf)

    prob_loss = label_conf * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=pred_prob)

    iou_loss = tf.reduce_mean(tf.reduce_sum(iou_loss, axis=[1,2,3]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3]))

    return iou_loss + conf_loss + prob_loss


def train(model):
    data = np.genfromtxt('TrainData.csv', delimiter=',', dtype=float)
    classes = np.array((data[:, -1]), dtype=int)
    print(len(classes), classes.max())
    onehot_classes = np.zeros((len(classes), classes.max()), dtype=int)
    # Classes start at index 1 in the file, not 0
    onehot_classes[np.arange(classes.size)-1, classes-1] = 1
    data = data[:, 0:-1]
    x_train, x_val, y_train, y_val = train_test_split(data, onehot_classes, test_size=0.2, shuffle=True)
    print(x_train.size, x_val.size, y_train.size, y_val.size)
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, validation_data=(x_val, y_val), epochs=NUM_EPOCHS, steps_per_epoch=100)

    

# Compile the model
if __name__ == "__main__":
    model.compile(optimizer='adam', loss='BinaryCrossentropy', metrics=['accuracy'])
    train(model)
    tf.saved_model.save(model, '')
    pred = model(test_input)
    print("Output size: ", tf.shape(pred))
