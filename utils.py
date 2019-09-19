import tensorflow as tf
import os, argparse, json

class TrainConfig:
    """ Class that have all training configurations to reproduce results """
    def __init__(self, parsed_args):
        """ Initialize TrainCofing object from parsed arguments by argparse.ArgumentParser """
        for k, v in parsed_args.__dict__.items():
            setattr(self, k, v)

    @classmethod
    def from_json(cls, path_to_json):
        """ Contruct a TrainConfig object from json_file
        Args:
            path_to_json: String, path to a .json file
        Returns:
            A TrainConfig object
        """
        with open(path_to_json) as f:
            kwargs = json.load(f)
        return cls(**kwargs)

    def save(self, save_path):
        """ Save a TrainConfig object as .json file """
        with open(save_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2, sort_keys=True)

    def __str__(self):
        json_string = json.dumps(self.__dict__, indent=2, sort_keys=True)
        return json_string


class TFObjectWrapper():
    """ Wrapper class for wrapping TensorFlow Object """
    def __init__(self):
        pass

    def get_current_session(self):
        """ Get a tf.Session object in current context manager """
        sess = tf.get_default_session()
        if sess is None:
            raise ValueError('This method must be called in tf.Session context manager')
        else:
            return sess

class TFScalarVariableWrapper(TFObjectWrapper):
    """ Wrapper for a non-trainable tensorflow scalar variable to checkpoint training state """
    def __init__(self, init_value, dtype, name):
        """ Initialize a TFScalarVariableWrapper object
        Args:
            init_value: A scalar, an initial value
            dtype: tf.dtypes.Dtype object
            name: String, name of this TF variable
        Returns:
            A TFScalarVariableWrapper object
        """
        super(TFScalarVariableWrapper, self).__init__()
        self.variable = tf.get_variable(name,
                                        shape=[],
                                        trainable=False,
                                        dtype=dtype,
                                        initializer=tf.constant_initializer(init_value))
        self.placeholder = tf.placeholder(dtype, shape=[], name='{}_pl'.format(name))
        self.assign_op = tf.assign(self.variable, self.placeholder)

    def eval(self):
        """ Get the value of a TF scalar variable """
        sess = self.get_current_session()
        return sess.run(self.variable)

    def assign(self, value):
        """ Assign a given value to this TF scalar variable  """
        sess = self.get_current_session()
        return sess.run(self.assign_op, feed_dict={self.placeholder:value})

    def init(self):
        """ Initialize this TF scalar variable """
        sess = self.get_current_session()
        sess.run(self.variable.initializer)

class TFSaverWrapper(TFObjectWrapper):
    """ Class to save and restore training states of the models implemented in tensorflow V1 """
    def __init__(self, save_dir):
        """ Initialize a TfSaverWrapper object
        Args:
            save_dir: String, directory path to create checkpoint file
        Returns:
            a TFSaverWrapper object
        """
        super(TFSaverWrapper, self).__init__()
        # Define attributes
        self.save_dir = save_dir
        self.latest_saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
        self.best_saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
        self.prefix_latest = os.path.join(save_dir, 'latest')
        self.prefix_best = os.path.join(save_dir, 'best')

    def checkpoint(self, is_best=False):
        """ Checkpoint model's current traning state: Both latest and best train state
        Args:
            is_best: Boolean, Whether current validation performace is the best
        Returns:
            None
        """
        # Create save directory if it does not exist
        os.makedirs(self.save_dir, exist_ok=True)

        # Save current training states
        sess = self.get_current_session()
        self.latest_saver.save(sess, self.prefix_latest)

        # Save the best training states
        if is_best:
            self.best_saver.save(sess, self.prefix_best)

    def restore_latest(self):
        """ Restore latest training state """
        sess = self.get_current_session()
        self.latest_saver.restore(sess, self.prefix_latest)

    def restore_best(self):
        """ Restore best training state """
        sess = self.get_current_session()
        self.best_saver.restore(sess, self.prefix_best)

def _test_TFScalarVariableWrapper():
    epoch = TFScalarVariableWrapper(0, tf.int64, 'epoch')

    with tf.Session() as sess:
        epoch.init()
        print(epoch.eval())
        epoch.assign(3)
        print(epoch.eval())
        epoch.assign(5)
        print(epoch.eval())
        epoch.init()
        print(epoch.eval())

def _test_TFSaverWrapper():
    save_dir = './foo'
    a = TFScalarVariableWrapper(0, tf.int64, 'a')
    b = TFScalarVariableWrapper(0, tf.int64, 'b')
    glob_init_op = tf.global_variables_initializer()
    saver = TFSaverWrapper(save_dir)

    with tf.Session() as sess:
        sess.run(glob_init_op)
        a.assign(3)
        b.assign(5)
        saver.checkpoint()
        print(saver.latest_checkpoint)
        #saver.restore_latest()
        #saver.restore_best()
        #print(a.eval())
        #print(b.eval())

def _test_TrainConfig():
    #train_config = TrainConfig(init_lr = 0.1,
    #                           epochs = 100,
    #                           hidden_units = [10, 20, 30])

    #train_config.save('sample.json')
    train_config = TrainConfig.from_json('./sample.json')
    #print(train_config.init_lr)
    #print(train_config.epochs)
    #print(train_config.hidden_units)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='gpu number', type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    #_test_TFScalarVariableWrapper()
    #_test_TFSaverWrapper()
    _test_TrainConfig()




