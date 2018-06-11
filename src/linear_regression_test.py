
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from base import SuperBasicTestCls

class LinearRegression(SuperBasicTestCls):
    def test_hello_world(self):
        with self.test_session() as sess:
            # y = W * x + b 이고 선형회귀분석을 사용할 것이다.

            num_points = 1000
            vectors_set = []

            for _ in range(num_points):
                x1 = np.random.normal(0.0, 0.55)
                y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
                vectors_set.append([x1, y1])

            x_data = [ v[0] for v in vectors_set ]
            y_data = [ v[1] for v in vectors_set ]

            # 우리는 이미 이 모델이 선형회귀라는 것을 알고 있으므로..
            # 게다가 이게 듀토리얼이라는 것도 이미 알고 있으므로 ㅎㅎ..

            # y_data = W * x_data = b 와 같은식으로 될것이다.
            # 원래 딥러닝수학에서는 y = px + q 이런식으로 설명을 했었던 것 같다.

            W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
            b = tf.Variable(tf.zeros([1]))

            y = W * x_data + b

            loss = tf.reduce_mean(tf.square(y - y_data))

            optimizer = tf.train.GradientDescentOptimizer(0.5)

            train = optimizer.minimize(loss)

            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            for _ in range(10):
                sess.run(train)

            print(sess.run(W), sess.run(b))


            plt.plot(x_data, y_data, 'ro')
            plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()
