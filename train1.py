import os
import tensorflow as tf
import numpy as np

# from cifar import networks, networkssate, data_provider_sattelite, data_provider
from Utils import get_init_vector
from cifar import networkssate as networks, data_provider_sattelite

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from grid_layout import create_mine_grid

tfgan = tf.contrib.gan
flags = tf.flags

flags.DEFINE_integer('batch_size', 64, 'The number of images in each batch.')

flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')

flags.DEFINE_string('train_log_dir', '/data/satellitegpu/train_log14',
                    'Directory where to write event logs.')

flags.DEFINE_string('dataset_dir', '/data/satellitegpu/', 'Location of data.')

flags.DEFINE_integer('max_number_of_steps', 100000,
                     'The maximum number of gradient steps.')

flags.DEFINE_integer(
    'ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

flags.DEFINE_integer(
    'task', 0,
    'The Task ID. This value is used when training with multiple workers to '
    'identify each worker.')

flags.DEFINE_boolean(
    'conditional', True,
    'If `True`, set up a conditional GAN. If False, it is unconditional.')

# Sync replicas flags.
flags.DEFINE_boolean(
    'use_sync_replicas', True,
    'If `True`, use sync replicas. Otherwise use async.')

flags.DEFINE_integer(
    'worker_replicas', 10,
    'The number of gradients to collect before updating params. Only used '
    'with sync replicas.')

flags.DEFINE_integer('backup_workers', 1, 'Number of workers to be kept as backup in the sync replicas case.')

flags.DEFINE_integer('generator_init_vector_size', 100, 'Generator initialization vector size')

FLAGS = flags.FLAGS


def main(_):
    if not tf.gfile.Exists(FLAGS.train_log_dir):
        tf.gfile.MakeDirs(FLAGS.train_log_dir)

    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
        # Force all input processing onto CPU in order to reserve the GPU for
        # the forward inference and back-propagation.
        with tf.name_scope('inputs'):
            with tf.device('/cpu:0'):
                images, one_hot_labels, _, _ = data_provider_sattelite.provide_data(
                    FLAGS.batch_size, FLAGS.dataset_dir)
            # images, one_hot_labels, _, _ = data_provider.provide_data(
            #        FLAGS.batch_size, FLAGS.dataset_dir)

        noise = get_init_vector(FLAGS.generator_init_vector_size, FLAGS.batch_size)

        generator_fn = networks.conditional_generator
        discriminator_fn = networks.conditional_discriminator
        generator_inputs = (noise, one_hot_labels)

        gan_model = tfgan.gan_model(
            generator_fn,
            discriminator_fn,
            real_data=images,
            generator_inputs=generator_inputs)
        tfgan.eval.add_gan_model_image_summaries(gan_model)

        # Get the GANLoss tuple. Use the selected GAN loss functions.
        # (joelshor): Put this block in `with tf.name_scope('loss'):` when
        # cl/171610946 goes into the opensource release.
        gan_loss = tfgan.gan_loss(gan_model,
                                  gradient_penalty_weight=1.0,
                                  add_summaries=True)

        # Get the GANTrain ops using the custom optimizers and optional
        # discriminator weight clipping.



        with tf.name_scope('train'):
            gen_lr, dis_lr = _learning_rate()
            gen_opt, dis_opt = _optimizer(gen_lr, dis_lr, FLAGS.use_sync_replicas)
            train_ops = tfgan.gan_train_ops(
                gan_model,
                gan_loss,
                generator_optimizer=gen_opt,
                discriminator_optimizer=dis_opt,
                summarize_gradients=True,
                colocate_gradients_with_ops=True,
                aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
            tf.summary.scalar('generator_lr', gen_lr)
            tf.summary.scalar('discriminator_lr', dis_lr)

        # Run the alternating training loop. Skip it if no steps should be taken
        # (used for graph construction tests).
        sync_hooks = ([gen_opt.make_session_run_hook(FLAGS.task == 0),
                       dis_opt.make_session_run_hook(FLAGS.task == 0)]
                      if FLAGS.use_sync_replicas else [])
        status_message = tf.string_join(
            ['Starting train step: ',
             tf.as_string(tf.train.get_or_create_global_step())],
            name='status_message')
        if FLAGS.max_number_of_steps == 0: return

        opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.7000)
        conf = tf.ConfigProto(gpu_options=opts)

        print("number of trainable variables")
        nn = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print(nn)

        tfgan.gan_train(
            train_ops,
            hooks=(
                    [tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps),
                     tf.train.LoggingTensorHook([status_message], every_n_iter=10)] +
                    sync_hooks),
            logdir=FLAGS.train_log_dir,
            master=FLAGS.master,
            is_chief=FLAGS.task == 0
        )



def _learning_rate():
    generator_lr = tf.train.exponential_decay(
        learning_rate=0.0001,
        global_step=tf.train.get_or_create_global_step(),
        decay_steps=100000,
        decay_rate=0.9,
        staircase=True)
    discriminator_lr = 0.001
    return generator_lr, discriminator_lr


def _optimizer(gen_lr, dis_lr, use_sync_replicas):
    """Get an optimizer, that's optionally synchronous."""
    # generator_opt = tf.train.RMSPropOptimizer(gen_lr, decay=.9, momentum=0.1)
    # discriminator_opt = tf.train.RMSPropOptimizer(dis_lr, decay=.95, momentum=0.1)
    generator_opt = tf.train.AdamOptimizer(gen_lr, beta1=0.5, beta2=0.999)
    discriminator_opt = tf.train.AdamOptimizer(dis_lr, beta1=0.5, beta2=0.999)

    def _make_sync(opt):
        return tf.train.SyncReplicasOptimizer(
            opt,
            replicas_to_aggregate=FLAGS.worker_replicas - FLAGS.backup_workers,
            total_num_replicas=FLAGS.worker_replicas)

    if use_sync_replicas:
        generator_opt = _make_sync(generator_opt)
        discriminator_opt = _make_sync(discriminator_opt)

    return generator_opt, discriminator_opt


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
