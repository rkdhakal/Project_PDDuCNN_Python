"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

from preprocessing import load_data

import os

import tensorflow as tf





tf.logging.set_verbosity(tf.logging.INFO)

NUM_OF_CLASSES = 6

DEFAULT_SIZE = 36



def cnn_model_fn(features, labels, mode):

    """Model function for CNN."""

    input_layer = tf.reshape(features["x"], [-1, DEFAULT_SIZE, DEFAULT_SIZE, 1])



    conv1 = tf.layers.conv2d(

        inputs=input_layer,

        filters=32,

        kernel_size=[5, 5],

        padding="same",

        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)



    conv2 = tf.layers.conv2d(

        inputs=pool1,

        filters=64,

        kernel_size=[5, 5],

        padding="same",

        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)



    conv3 = tf.layers.conv2d(

        inputs=pool2,

        filters=128,

        kernel_size=[5, 5],

        padding="same",

        activation=tf.nn.relu)

    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)



    pool2_flat = tf.reshape(pool3, [-1, 4 * 4 * 128])



    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    #dense1 = tf.layers.dense(inputs=dense, units=256, activation=tf.nn.sigmoid)

    # Add dropout operation; 0.6 probability that element will be kept

    dropout = tf.layers.dropout(

        inputs=dense, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=NUM_OF_CLASSES)



    predictions = {

        # Generate predictions (for PREDICT and EVAL mode)

        "classes": tf.argmax(input=logits, axis=1),

        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the

        # `logging_hook`.

        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")

    }



    # 4. Create export outputs

    export_outputs = {"predicted": tf.estimator.export.PredictOutput(predictions)}

    if mode == tf.estimator.ModeKeys.PREDICT:

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)



    # Calculate Loss (for both TRAIN and EVAL modes)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)

    if mode == tf.estimator.ModeKeys.TRAIN:

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        train_op = optimizer.minimize(

            loss=loss,

            global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)



    # Add evaluation metrics (for EVAL mode)

    eval_metric_ops = {

        "accuracy": tf.metrics.accuracy(

            labels=labels, predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(

        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)





def main(unused_argv):



    ROOT_PATH = "."  # Denotes the current working directory

    TRAIN_DATA_DIRECTORY = os.path.join(ROOT_PATH, "/root/leaf_image/DATA/training")

    TEST_DATA_DIRECTORY = os.path.join(ROOT_PATH, "/root/leaf_image/DATA/testing")



    train_data, train_labels = load_data(TRAIN_DATA_DIRECTORY)

    eval_data, eval_labels = load_data(TEST_DATA_DIRECTORY)



    # Create the Estimator

    mnist_classifier = tf.estimator.Estimator(

        model_fn=cnn_model_fn, model_dir="./tmp/model")



    # Set up logging for predictions

    # Log the values in the "Softmax" tensor with label "probabilities"

    tensors_to_log = {"probabilities": "softmax_tensor"}

    logging_hook = tf.train.LoggingTensorHook(

        tensors=tensors_to_log, every_n_iter=50)



    # Train the model

    train_input_fn = tf.estimator.inputs.numpy_input_fn(

        x={"x": train_data},

        y=train_labels,

        batch_size=100,

        num_epochs=None,

        shuffle=True)

    mnist_classifier.train(

        input_fn=train_input_fn,

        steps=100,

        hooks=[logging_hook])



    def serving_input_receiver_fn():

        """Build the serving inputs."""



        inputs = {"x": tf.placeholder(shape=[1, DEFAULT_SIZE, DEFAULT_SIZE, 1], dtype=tf.float32)}

        return tf.estimator.export.ServingInputReceiver(inputs, inputs)



    export_dir = mnist_classifier.export_savedmodel(

        export_dir_base="./model_saved/",

        serving_input_receiver_fn=serving_input_receiver_fn)



    # Evaluate the model and print results

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(

        x={"x": eval_data},

        y=eval_labels,

        num_epochs=2,

        shuffle=False)

    print(eval_input_fn)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)

    print(eval_results)



    predict_input_fn = tf.estimator.inputs.numpy_input_fn(

        x={"x": train_data[0]},

        shuffle=False)

    prediction_results = mnist_classifier.predict(predict_input_fn)

    for i in prediction_results:

        print(i)

        print(i['classes'])

if __name__ == "__main__":

    tf.app.run()