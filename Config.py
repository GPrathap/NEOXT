import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 200, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "First momentum term of adam [0.5]")
flags.DEFINE_float("beta2", 0.5, "Second momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 64, "The number of batch images [64]")
flags.DEFINE_integer("image_size", 64, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("sample_step", 500, "The interval of generating sample. [500]")
flags.DEFINE_integer("save_step", 50, "The interval of saving checkpoints. [500]")
flags.DEFINE_string("checkpoint_dir", "/data/checkpoint91", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "/data/samples91", "Directory name to save the image samples [samples]")
flags.DEFINE_string("feature_dir", "/data/features91", "Directory name to save features")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_string('dataset_storage_location', '/data/neotx', 'Location of image store')
#flags.DEFINE_string('dataset_name', 'pattennet', 'Data set name')
#flags.DEFINE_string('dataset_name', 'nwpu_resisc45', 'Data set name')
#flags.DEFINE_string('dataset_name', 'brazilian_coffee_scenes', 'Data set name')
flags.DEFINE_string('dataset_name', 'ucdataset', 'Data set name')
FLAGS = flags.FLAGS