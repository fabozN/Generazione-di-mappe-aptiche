import absl.flags as flags
import torch

FLAGS = flags.FLAGS

# System variables
flags.DEFINE_integer('seed', 42, 'Random seed')
flags.DEFINE_string('device', 'cuda' if torch.cuda.is_available() else 'cpu', 'Device')
flags.DEFINE_string('idle_device', 'cpu', 'Idle device')
flags.DEFINE_boolean('test', False, 'Test mode')

# Wandb initialization
flags.DEFINE_boolean('wandb', False, 'Use wandb')
flags.DEFINE_string('wandb_project', 'touch_diffusion_model', 'Wandb project')
flags.DEFINE_string('wandb_group', 'pre_training_on_tarf_from_scratch', 'Wandb group')

# Data parameters
flags.DEFINE_string('data_dir', 'dataset/tripletta/', 'Path to data')
flags.DEFINE_string('use_loader', 'nocs', 'Loader to use --> Keep nocs')
flags.DEFINE_list('cond_options', ['rgb','nocs'], 'Conditioning options. Options: rgb, nocs. Delete the ones you do not want to use')
flags.DEFINE_integer('image_size', 256, 'Image size')
flags.DEFINE_float('split_train', 0.8, 'Train split')
flags.DEFINE_integer('breaking_point', 0, 'Breaking point')
flags.DEFINE_integer('test_breaking_point', 0, 'Test breaking point')

# Model initialization
flags.DEFINE_string('pretrained', 'touch', 'Pretrained model to load --> keep touch')
flags.DEFINE_string('pre_trained_model', 'nothing', 'keep nothing')
flags.DEFINE_string('conditioning', 'concat', 'Conditioning type --> concat, attention or uncondition')
flags.DEFINE_string('cond_type', 'nocs', 'Conditioning type --> keep nocs')
flags.DEFINE_string('model_name', 'resnet50', 'Model name')

# Diffusion parameters
flags.DEFINE_string('sampler_name', 'ddpm', 'Diffusion type')
flags.DEFINE_integer('num_training_steps', 300, 'Number of training timesteps')
flags.DEFINE_integer('num_inference_steps', 10, 'Number of inference timesteps')

# Training parameters
flags.DEFINE_integer('epochs', 10, 'Number of epochs')
flags.DEFINE_integer('batch_size', 2, 'Batch size')
flags.DEFINE_float('lr', 1e-4, 'Learning rate')
flags.DEFINE_bool('scheduling_lr', False, 'Scheduling learning rate')

# Logging parameters
flags.DEFINE_boolean('log', True, 'Log')
flags.DEFINE_integer('log_interval', 10, 'Log interval')
flags.DEFINE_integer('save_interval', 100, 'Save interval')