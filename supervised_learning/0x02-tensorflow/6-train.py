#!/usr/bin/env python3
"""
Builds, trains, and saves a neural network classifier
"""


import tensorflow as tf
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier
    Args:
        - X_train: is a numpy.ndarray containing the training input data.
        - Y_train: is a numpy.ndarray containing the training labels.
        - X_valid: is a numpy.ndarray containing the validation input data.
        - Y_valid: is a numpy.ndarray containing the validation labels.
        - layer_sizes: is a list containing the number of nodes in each layer
          of the network.
        - activations: is a list containing the activation functions for each
          layer of the network.
        - alpha: is the learning rate.
        - iterations: is the number of iterations to train over.
        - save_path: designates where to save the model.
    Return:
        The path where the model was saved.
    """
    step = 100

    # Placeholders x, y
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    # Tensors
    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    # Operation
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)

    # Setting global initializer
    init_op = tf.global_variables_initializer()

    # Setting the operation to save and restore all variables
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)

        for i in range(iterations + 1):
            t_cost, t_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_train, y: Y_train})

            v_cost, v_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_valid, y: Y_valid})

            if i % step == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(t_cost))
                print("\tTraining Accuracy: {}".format(t_accuracy))
                print("\tValidation Cost: {}".format(v_cost))
                print("\tValidation Accuracy: {}".format(v_accuracy))

            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        return saver.save(sess, save_path)
