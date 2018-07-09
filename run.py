"""Main script."""


import json
import os
import sys
import argparse
import six
import yaml
from collections import namedtuple

import tensorflow as tf

from runner import Runner
from opennmt.config import load_model, load_config

def _prefix_paths(prefix, paths):
  """Recursively prefix paths.

  Args:
    prefix: The prefix to apply.
    data: A dict of relative paths.

  Returns:
    The updated dict.
  """
  if isinstance(paths, dict):
    for key, path in six.iteritems(paths):
      paths[key] = _prefix_paths(prefix, path)
    return paths
  else:
    path = paths
    new_path = os.path.join(prefix, path)
    if os.path.isfile(new_path):
      return new_path
    else:
      return path

def main(run_config):

  args = yaml.load(open(run_config))
  args = namedtuple("Execution_Config", args.keys())(*args.values())

  tf.logging.set_verbosity(getattr(tf.logging, args.log_level))

  # Setup cluster if defined.
  if args.chief_host:
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": {
            "chief": [args.chief_host],
            "worker": args.worker_hosts.split(","),
            "ps": args.ps_hosts.split(",")
        },
        "task": {
            "type": args.task_type,
            "index": args.task_index
        }
    })

  # Load and merge run configurations.
  config = load_config(args.config)
  if args.run_dir:
    config["model_dir"] = os.path.join(args.run_dir, config["model_dir"])
  if args.data_dir:
    config["data"] = _prefix_paths(args.data_dir, config["data"])

  if not os.path.isdir(config["model_dir"]):
    tf.logging.info("Creating model directory %s", config["model_dir"])
    os.makedirs(config["model_dir"])

  model = load_model(config["model_dir"], model_file=args.model, model_name=args.model_type)
  session_config = tf.ConfigProto(
      intra_op_parallelism_threads=args.intra_op_parallelism_threads,
      inter_op_parallelism_threads=args.inter_op_parallelism_threads)
  runner = Runner(
      model,
      config,
      seed=args.seed,
      num_devices=args.num_gpus,
      gpu_allow_growth=args.gpu_allow_growth,
      session_config=session_config)

  if args.run == "train_and_eval":
    runner.train_and_evaluate(checkpoint_path=args.checkpoint_path,
                              eval_train=args.evaluate_train)
  elif args.run == "train":
    runner.train()
  elif args.run == "eval":
    runner.evaluate(checkpoint_path=args.checkpoint_path, 
                    eval_train=args.evaluate_train)
  elif args.run == "infer":
    if not args.features_file:
      print('ERROR: features_file is required for inference.')
      raise SystemExit
    elif len(args.features_file) == 1:
      args.features_file = args.features_file[0]
    runner.infer(
        args.features_file,
        predictions_file=args.predictions_file,
        checkpoint_path=args.checkpoint_path,
        log_time=args.log_prediction_time)
  elif args.run == "export":
    runner.export(checkpoint_path=args.checkpoint_path)
  elif args.run == "score":
    if not args.features_file:
      print('ERROR: features_file is required for scoring.')
      raise SystemExit
    if not args.predictions_file:
      print('ERROR: predictions_file is required for scoring.')
      raise SystemExit
    runner.score(
        args.features_file,
        args.predictions_file,
        checkpoint_path=args.checkpoint_path)

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('run_config',
                      help='Path to .yml file with the configurations for the run.')
  args = parser.parse_args()

  main(args.run_config)