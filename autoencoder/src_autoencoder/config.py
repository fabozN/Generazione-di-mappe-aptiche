import absl.flags as flags

FLAGS = flags.FLAGS

flags.DEFINE_string('device', 'cuda:0', 'Device')
flags.DEFINE_integer('seed', 42, 'Random seed')
flags.DEFINE_integer('batch_size', 4, 'Batch size')
flags.DEFINE_integer('epochs', 30, 'Number of epochs')
flags.DEFINE_float('lr', 1e-6, 'Learning rate')
#flags.DEFINE_string('data_dir', './tarf_dataset', 'Data directory')
flags.DEFINE_string('data_dir', '/home/fabo/codiceWSL/ObjectFolder/demo', 'Data directory')
flags.DEFINE_integer('image_size', 256, 'Image size')
flags.DEFINE_integer('breaking_point', 0, 'Breaking point for data loading --> 0 means no breaking point')
flags.DEFINE_integer('test_breaking_point', 0, 'Breaking point for test data loading --> 0 means no breaking point')
flags.DEFINE_boolean('pretrained', False, 'Use pretrained model')
flags.DEFINE_integer('num_workers', 8, 'Number of workers for data loader')

# Logging parameters
flags.DEFINE_boolean('log', True, 'Log images and data or just test code')
flags.DEFINE_boolean('wandb', False, 'Use wandb for logging')
flags.DEFINE_string('wandb_project', 'touch_autoencoder', 'wandb project name')
flags.DEFINE_string('wandb_group', 'res256', 'wandb group name')
