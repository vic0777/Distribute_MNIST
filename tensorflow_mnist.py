#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import time
import tensorflow as tf
import os
from tensorflow.python.framework import dtypes

flags = tf.app.flags

# 定义数据路径
flags.DEFINE_string('dataset_dir', os.getenv('DATASET_DIR', '`pwd`/dataset'), 'Directory  for storing mnist data')
flags.DEFINE_string('result_dir', os.getenv('RESULT_DIR', '`pwd`/result'), 'Directory  for storing log data')

# 训练参数
flags.DEFINE_integer('train_steps', os.getenv('TRAIN_STEPS', 10000), 'Number of training steps to perform')
flags.DEFINE_integer('batch_size', os.getenv('BATCH_SIZE', 100), 'Training batch size ')
flags.DEFINE_float('learning_rate', os.getenv('LEARNING_RATE', 0.001), 'Learning rate')
flags.DEFINE_float('save_checkpoint_steps', os.getenv('SAVE_CHECKPOINT_STEPS', None),
                   'in number of global steps, that a checkpoint is saved using a default checkpoint saver')
flags.DEFINE_float('save_checkpoint_secs', os.getenv('SAVE_CHECKPOINT_SECS', 10),
                   'in seconds, that a checkpoint is saved using a default checkpoint saver')

# 定义分布式参数
# 参数服务器parameter server节点
flags.DEFINE_string('ps_hosts', os.getenv('PS_HOSTS', '127.0.0.1:22220'), 'Comma-separated list of hostname:port pairs')
# worker节点
flags.DEFINE_string('worker_hosts', os.getenv('WORKER_HOSTS', '127.0.0.1:22221'),
                    'Comma-separated list of hostname:port pairs')
# 设置job name参数
flags.DEFINE_string('job_name', os.getenv('JOB_NAME', 'worker'), 'job name: worker or ps')
# 设置任务的索引
flags.DEFINE_integer('task_index', os.getenv('TASK_INDEX', '0'), 'Index of task within the job')
# 选择异步并行，同步并行
flags.DEFINE_integer("is_sync", os.getenv('IS_SYNC', '0'), "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")

flags.DEFINE_string('log_level', os.getenv('LOG_LEVEL', 'INFO'), 'log level')

FLAGS = flags.FLAGS


def read_data_sets():
    from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets as _read_data_sets
    return _read_data_sets(FLAGS.dataset_dir, one_hot=True)


def model(images):
    """Define a simple mnist classifier"""
    net = tf.layers.dense(images, 500, activation=tf.nn.relu)
    net = tf.layers.dense(net, 500, activation=tf.nn.relu)
    net = tf.layers.dense(net, 10, activation=None)
    return net


def main(_):
    tf.logging.set_verbosity(FLAGS.log_level)

    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name !')
    else:
        print('job_name : %s' % FLAGS.job_name)
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index!')
    else:
        tf.logging.info('task_index : %d' % FLAGS.task_index)

    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # create the cluster configured by `ps_hosts' and 'worker_hosts'
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    config = tf.ConfigProto(allow_soft_placement=True)

    # 最多占gpu资源的70%
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # 开始不会给tensorflow全部gpu资源 而是按需增加
    config.gpu_options.allow_growth = True
    # create a server for local task
    server = tf.train.Server(cluster, job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index, config=config)
    if FLAGS.job_name == "ps":
        server.join()  # ps hosts only join
    elif FLAGS.job_name == "worker":
        # workers perform the operation
        # ps_strategy = tf.contrib.training.GreedyLoadBalancingStrategy(FLAGS.num_ps)

        # Note: tf.train.replica_device_setter automatically place the paramters (Variables)
        # on the ps hosts (default placement strategy:  round-robin over all ps hosts, and also
        # place multi copies of operations to each worker host
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % (FLAGS.task_index),
                                                      cluster=cluster)):
            # load mnist dataset
            mnist = read_data_sets()

            # the model
            images = tf.placeholder(tf.float32, [None, 784])
            labels = tf.placeholder(tf.int32, [None, 10])

            logits = model(images)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

            # The StopAtStepHook handles stopping after running given steps.
            hooks = [tf.train.StopAtStepHook(last_step=FLAGS.train_steps)]

            global_step = tf.train.get_or_create_global_step()
            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

            if FLAGS.is_sync:
                # asynchronous training
                # use tf.train.SyncReplicasOptimizer wrap optimizer
                # ref: https://www.tensorflow.org/api_docs/python/tf/train/SyncReplicasOptimizer
                optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=FLAGS.num_workers,
                                                           total_num_replicas=FLAGS.num_workers)
                # create the hook which handles initialization and queues
                hooks.append(optimizer.make_session_run_hook((FLAGS.task_index == 0)))

            train_op = optimizer.minimize(loss, global_step=global_step,
                                          aggregation_method=tf.AggregationMethod.ADD_N)

            # The MonitoredTrainingSession takes care of session initialization,
            # restoring from a checkpoint, saving to a checkpoint, and closing when done
            # or an error occurs.
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   is_chief=(FLAGS.task_index == 0),
                                                   checkpoint_dir=FLAGS.result_dir,
                                                   save_checkpoint_steps=FLAGS.save_checkpoint_steps,
                                                   save_checkpoint_secs=FLAGS.save_checkpoint_secs,
                                                   hooks=hooks) as mon_sess:

                while not mon_sess.should_stop():
                    # mon_sess.run handles AbortedError in case of preempted PS.
                    img_batch, label_batch = mnist.train.next_batch(FLAGS.batch_size)
                    _, ls, step = mon_sess.run([train_op, loss, global_step],
                                               feed_dict={images: img_batch, labels: label_batch})

                    tf.logging.info("Train step %d, loss: %f" % (step, ls))


if __name__ == '__main__':
    tf.app.run()
