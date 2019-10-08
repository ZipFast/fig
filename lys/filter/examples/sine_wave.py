r'''
==================================
Kalman Filter tracking a sine wave
==================================

This example shows how to use the Kalman Filter for state estimation.

In this example, we generate a fake target trajectory using a sine wave.
Instead of observing those positions exactly, we observe the position plus some
random noise.  We then use a Kalman Filter to estimate the velocity of the
system as well.

The figure drawn illustrates the observations, and the position and velocity
estimates predicted by the Kalman Smoother.
'''
import numpy as np
import tensorflow as tf
from .tfkalman import filters







def kalmanfilter(observations,n_timesteps):

    if len(observations)==0:
        return observations

    # rnd = np.random.RandomState(0)  # 为随机数生成器，确保生成一致的随机数
    # # generate a noisy sine wave to act as our fake observations
    # n_timesteps = 200
    # x_axis = np.linspace(0, 5 * np.pi, n_timesteps)  # 在指定的间隔内返回均匀间隔的数字。
    # observations = 20 * (np.sin(x_axis) + 0.5 * rnd.randn(n_timesteps))  # 计算正弦值并加入噪声，然后放大20倍
    #  m: int - measurement size
    # n: int - state size
    # l: int - control input size
    n = 1
    m = 1
    l = 1

    # X : State at step k - 1 (apriori) [n]   #先验值、初始值。用于估计下一个值
    # P : state error covariance at step k - 1 (apriori) [n, n]    先验状态误差协方差

    # A : transition matrix [n, n]   过度矩阵
    # Q : The process noise covariance matrix.  过程噪声协方差矩阵(测量噪声)
    # B : The input effect matrix.  输入效果矩阵
    # U : The control input.    控制输入
    x = np.ones([1, 1])

    A = np.ones([1, 1])

    B = np.zeros([1, 1])

    P = np.ones([1, 1])

    Q = np.array([[0.0003]])

    H = np.ones([1, 1])

    u = np.zeros([1, 1])

    R = np.array([[0.1]])

    predictions = []
    with tf.Session() as sess:
        kf = filters.KalmanFilter(x=x, A=A, B=B, P=P, Q=Q, H=H)
        predict = kf.predict()
        correct = kf.correct()
        tf.global_variables_initializer().run()
        for i in range(0, n_timesteps):
            x_pred, _ = sess.run(predict, feed_dict={kf.u: u})
            # print x_pred, p_pred
            predictions.append(x_pred[0, 0])
            sess.run(correct, feed_dict={kf.z: np.array([[observations[i]]]), kf.R: R})
    return np.array(predictions)





