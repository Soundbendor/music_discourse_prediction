import configparser
import neptune.new as neptune 
import tensorflow as tf

from typing import Tuple
from neptune.new.integrations.tensorflow_keras import NeptuneCallback


def get_num_gpus() -> int:
    return len(tf.config.list_physical_devices('GPU'))

def _process_api_key(f_key: str) -> configparser.ConfigParser:
    api_key = configparser.ConfigParser()
    api_key.read(f_key)
    return api_key

def tf_config() -> Tuple[tf.distribute.Strategy, tf.data.Options]: 
    print(f"Num GPUs Available: {get_num_gpus()}")

    configproto = tf.compat.v1.ConfigProto() 
    configproto.gpu_options.allow_growth = True
    configproto.gpu_options.polling_inactive_delay_msecs = 10
    sess = tf.compat.v1.Session(config=configproto) 
    tf.compat.v1.keras.backend.set_session(sess)

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    tf.debugging.set_log_device_placement(True)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    return strategy, options

def init_neptune(cfg: str):
    creds = _process_api_key(cfg)
    runtime = neptune.init(project=creds['CLIENT_INFO']['project_id'],
                        api_token=creds['CLIENT_INFO']['api_token'])
    return NeptuneCallback(run=runtime, base_namespace='metrics')