import absl.flags as flags

FLAGS = flags.FLAGS

flags.DEFINE_string('device', 'cuda', 'Device')
flags.DEFINE_integer('seed', 42, 'Random seed')
flags.DEFINE_integer('batch_size', 8, 'Batch size')
flags.DEFINE_string('data_dir', '/home/fabo/codiceWSL/ObjectFolder/demo', 'Data directory')
flags.DEFINE_integer('image_size', 256, 'Image size')
flags.DEFINE_integer('breaking_point', 2000, 'Breaking point for data loading --> 0 means no breaking point')
flags.DEFINE_integer('test_breaking_point', 2000, 'Breaking point for test data loading --> 0 means no breaking point')
flags.DEFINE_integer('save_any', 10, 'Save some sample images each n batches')
