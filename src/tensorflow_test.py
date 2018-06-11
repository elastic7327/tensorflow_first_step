#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf

from base import SuperBasicTestCls

class BasicTest(SuperBasicTestCls):
    def test_hello_world(self):
        with self.test_session() as sess:
            a = tf.placeholder("float")
            b = tf.placeholder("float")
            y = tf.multiply(a, b)
            # 개인적인생각으로는 이런식으로 나중에 인풋값을 가짐으로서
            # 미분을 할때에 정말 정말 편할 것 같다.
            sess.run(y, feed_dict={a: 3, b: 3})
