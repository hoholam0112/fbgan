import tensorflow as tf
import os, argparse, json

class TrainConfig:
    """ Class that have all training configurations to reproduce results """
    def __init__(self, **kwargs):
        """ Initialize TrainCofing object from parsed arguments by argparse.ArgumentParser """
        for k, v in kwargs.items():
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
    def __init__(self, save_dir, var_list):
        """ Initialize a TfSaverWrapper object

        Args:
            save_dir: String, directory path to create checkpoint file
            var_list: List of tf.Varible objects. Variables to be restored

        Returns:
            a TFSaverWrapper object
        """
        super(TFSaverWrapper, self).__init__()
        # Define attributes
        self.save_dir = save_dir
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=1)

    def checkpoint(self):
        """ Checkpoint model's current traning state """
        # Create save directory if it does not exist
        os.makedirs(self.save_dir, exist_ok=True)
        # Save current training states
        sess = self.get_current_session()
        self.saver.save(sess, os.path.join(self.save_dir, 'ckpt'))

    def restore(self):
        """ Restore latest training state """
        sess = self.get_current_session()
        self.saver.restore(sess, os.path.join(self.save_dir, 'ckpt'))

