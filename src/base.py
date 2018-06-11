#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


class SuperBasicTestCls(tf.test.TestCase):

  def testSquare(self):
    with self.test_session():
      x = tf.square([2, 3])
      self.assertAllEqual(x.eval(), [4, 9])
