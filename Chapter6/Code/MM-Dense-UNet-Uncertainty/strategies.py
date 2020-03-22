import torch
import torch.nn as nn

import pytu
from pytu.sacred import check_object
from pytu.strategies import utils
from pytu.strategies.base_strategy import BaseStrategy
from pytu.misc.dict_utils import unravel_dict


class SimpleStrategy(BaseStrategy):
    """ Simple forward/backward based strategy.

    Given inputs and targets, this strategy executes a forward
    pass on the model to retrieve a prediction. Then, it computes
    the gradients based on a given loss, w.r.t the target. Model
    is then updated using the provided optimizer.

    :param model: The model to update
    :type model: :mod:`torch.nn.Module`
    :param optimizer: The optimizer to use for updating the model
    :type optimizer: :mod:`torch.optim.Optimizer`
    :param loss: Loss function to use to compute the gradients
    :type loss: :mod:`torch.nn._Loss`
    :param cuda: Whether GPU should be used. Default is ``True``
    :type cuda: bool, optional
    :param multi_gpu: Whether multiple GPUs should be used.
     Default is ``False``
    :type multi_gpu: bool, optional
    """
    @pytu.sacred.register
    def __init__(self, model, optimizer=None, loss=None,
                 cuda=True, multi_gpu=False):
        # type: (object, object, object, object, object) -> object
        super().__init__()
        self.cuda = cuda
        self.model = model.cuda() if self.cuda else model
        self.multi_gpu = multi_gpu
        if multi_gpu:
            self.model = nn.DataParallel(self.model)
        self.optimizer = optimizer
        self.loss = check_object(loss)
        if self.loss is None:
            print('Warning: You should not create a strategy with loss=None.')
        self.metadata = {
            'Optimizer': type(self.optimizer).__name__,
            'Loss': type(self.loss).__name__
        }
        self.model_callbacks = []
        self.validation_callbacks = []
        self.train_callbacks = []

    def read_data(self, data):
        """Defines the way data is being read.

        Default behavior is just returning the data.
        This is the function that should be updated if any preprocessing
        has to happend to the data before being fed to the model.

        :param data: The data is being returned by the data loader
        :type data: iterable
        :return: The data fed to the model
        :rtype: iterable

        .. note:: This is not where perturbations should be added. See
         :mod:`pytu.data.PerturbationDataset`
        """
        return data

    def add_model_callback(self, callback, **kwargs):
        """Add a callback to apply on the model or the optimizer.

        After each epoch, callbacks may be applied to the model, or
        the optimizer. You can also forward keyword arguments to
        the given callback.

        Your callback should at least provide the following signature,
        but more keyword arguments may be provided; they will be forwarded::

            def my_callback(model, optimizer, validation_loss, epoch, step):
                  # Do something

        Example::

            def print_epoch_callback(model,
                                     optimizer,
                                     validation_loss,
                                     epoch,
                                     step,
                                     prefix='This is epoch'):
                print(prefix, epoch)

         >>> strat = SimpleStrategy(...)
         >>> strat.add_model_callback(print_epoch_callback, prefix='epoch:')

        :param callback: The function to apply
        :type callback: Callable
        """
        func = lambda *args: callback(*args, **kwargs)
        self.model_callbacks.append(func)

    def add_validation_callback(self, callback, **kwargs):
        """Add a callback to apply during the validation step.

        After each epoch, callbacks may be applied to the predictions.
        You can also forward keyword arguments to the given callback.
        Your callback can also return a dict with any metric you would
        like to track. Please note that if you want a real metric with
        different aggregation methods (mean, percentile...) you should
        use metrics from :mod:`pytu.iterators`.

        Your callback should at least provide the following signature,
        but more keyword arguments may be provided; they will
        be forwarded::

            def my_callback(input, prediction, target, epoch, step):
                  # Do something
                  return {'value_to_track': 10}

        Example::

            def save_images(input, prediction, target,
                            epoch, step, opath='images'):
                print('Epoch', epoch, 'validation_step', step)
                # Save images in `opath`

         >>> strat = SimpleStrategy(...)
         >>> strat.add_validation_callback(save_images, opath='results')

        :param callback: The function to apply
        :type callback: Callable
        """
        func = lambda *args: callback(*args, **kwargs)
        self.validation_callbacks.append(func)

    def add_train_callback(self, callback, **kwargs):
        """Add a callback to apply during the train step.

        After each training step, callbacks may be applied to the predictions.
        You can also forward keyword arguments to the given callback.

        Your callback should at least provide the following signature,
        but more keyword arguments may be provided; they will
        be forwarded::

            def my_callback(input, prediction, target, err, idx):
                  # Do something

        Example::

            def log_train_loss(input, prediction, target, err, idx, logger=None, log_every=5):
                if idx % log_every == 0:
                    logger.scalar_summary('train/loss', err['loss'], idx)

         >>> strat = SimpleStrategy(...)
         >>> strat.add_train_callback(log_train_loss, logger=MyLogger())

        :param callback: The function to apply
        :type callback: Callable
        """
        func = lambda *args: callback(*args, **kwargs)
        self.train_callbacks.append(func)

    def load_weights(self, path):
        """Loads weights to the model.

        This function loads the ``state_dict`` saved in ``path`` into
        the model. This function shall work no matter if the weights
        were from cpu, single gpu or multi gpu, weights will be loaded
        according to the current strategy.

        :param path: The path to the weight file
        :type path: str
        .. note:: Even though this function works with any kind of
         presaved models, keep in mind that for portability reasons,
         weights should not be saved in multi gpu.
        """
        if self.multi_gpu:
            try:
                self.model.load_state_dict(torch.load(path))
            except:
                self.model.module.load_state_dict(torch.load(path))
        else:
            self.model.load_state_dict(torch.load(path))

    def load_model(self, path):
        """Loads a model.

        This function loads the serialized model saved in ``path``.
        This function shall work no matter if the model
        was from cpu, single gpu or multi gpu, model will be loaded
        according to the current strategy.

        :param path: The path to the model file
        :type path: str
        .. note:: When saving a model, only parameters are serialized.
         Keep in mind that the network definition should always be
         available when trying to load a model. It shall not be an issue
         when dealing with :mod:`pytu.networks` but may be of concern
         when using your own networks.
        """
        self.model = torch.load(path)
        if self.multi_gpu and type(self.model) != nn.DataParallel:
            self.model = nn.DataParallel(self.model)

    def save_weights(self, path):
        """Saves weights of the current model.

        This function saves the ``state_dict`` into ``path``.
        It will export a cpu / single gpu version of the weights, for
        portability purposed.

        :param path: The path of the output weight file.
        :type path: str
        """
        if self.multi_gpu:
            torch.save(self.model.module.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)

    def save_optimizer(self, path):
        """Saves current state of optimizer.

        This function saves the ``state_dict`` of the optimizer into ``path``.

        :param path: The path to the output state file.
        :type path: str
        """
        torch.save(self.optimizer.state_dict(), path)

    def load_optimizer(self, path):
        """Loads optimizer.

        This function retrieves the state of a previously saved optimizer.

        :param path: The path to the saved optimizer.
        :type path: str
        """
        self.optimizer.load_state_dict(torch.load(path))

    def save_model(self, path):
        """Serialize the current model.

        This function saves the current model into ``path``.
        It will export a cpu / single gpu version of the model, for
        portability purposed.

        :param path: The path of the output weight file.
        :type path: str
        .. note:: When saving a model, only parameters are serialized.
         Keep in mind that the network definition should always be
         available when trying to load a model. It shall not be an issue
         when dealing with :mod:`pytu.networks` but may be of concern
         when using your own networks.
        """
        if self.multi_gpu:
            torch.save(self.model.module, path)
        else:
            torch.save(self.model, path)

    def update(self, clear_grad=True):
        """ Updates the weights of the current model based on the current
        gradients.

        It basically calls a step on the optimizer.

        :param clear_grad: Whether gradients should be cleared
         after update. Default is ``True``
        :type clear_grad: bool, optional
        """
        self.optimizer.step()
        if clear_grad:
            self.clear_grad()

    def clear_grad(self):
        """ Clears the gradients of the model.

        You may want to override this function if you want to apply
        a specific function before clearing the gradients.
        """
        self.optimizer.zero_grad()

    def get_loss(self, ins, tgs, idx,
                 freeze=False,
                 pred=False):
        """Given a input, execute a forward pass and computes the loss,
        w.r.t the target.

        You can choose to freeze the network when doing the forward pass,
        gradients variables will not be allocated and forward will be faster
        and will consume less memory. This is useful for validation or
        inference.

        You can also choose to return the prediction. This prevents useless
        forward when doing validation.

        :param ins: the input for the forward pass
        :type ins: Tensor
        :param freeze: Whether network should be frozen. Default is ``False``
        :type freeze: bool, optional
        :param pred: Whether prediction should be returned. Default is ``False``
        :type pred: bool, optional
        :return: The loss w.r.t the target, and optionally the prediction
        """
        prediction = self.predict(ins, freeze=freeze)
        err = self.loss(prediction, tgs, idx)
        return err if not pred else (err, prediction)

    def step(self, data, idx):
        """Executes a forward / backward step.

        It basically reads the data, executes a forward pass,
        computes the loss, and executes a backward pass computing
        the gradients w.r.t the target and updates the weights accordingly.

        :param data: The data to use (input, target)
        :type data: iterable
        :param idx: The current step
        :type idx: int
        :return: The value of the loss, rounded with 6 decimals
        :rtype: dict
        """
        ins, tgs = self.read_data(data)
        if len(self.train_callbacks) > 0:
            err, pred = self.get_loss(ins, tgs, idx, pred=True)
            for callback in self.train_callbacks:
                callback(ins, pred, tgs, err, idx)
        else:
            err = self.get_loss(ins, tgs, idx, pred=False)
        err.backward()
        self.update()

        return {'loss': round(err.item(), 6)}

    def validation_step(self, data, epoch, step,
                        evalmode=True):
        """Computes a validation step on a given pair of input
        and targets.

        This function is called by the trainer after each epoch.
        This is where the callbacks are called.

        :param data: The data to use (input, target)
        :type data: iterable
        :param epoch: The current epoch
        :type epoch: int
        :param step: The current validation step
        :type step: int
        :param evalmode: Whether network should be in evaluation mode.
         Default is ``True``
        :type evalmode: bool, optional
        :return: The input, the prediction, the target and the loss
        :rtype: Tensor, Tensor, Tensor, dict

        .. note:: When network is in evaluation mode, dropout is not used,
         and normalization layers do not compute statistics of current batch.
         Only training statistics are used.
        """
        if evalmode:
            self.model.eval()
        ins, tgs = self.read_data(data)
        err, pred = self.get_loss(ins, tgs, idx=-1, pred=True, freeze=True)
        res = {'validation_loss': err.item()}
        for callback in self.validation_callbacks:
            res_ = callback(ins.cpu(), pred.cpu(),
                            tgs.cpu(), epoch, step)
            if type(res_) == dict:
                res.update(res_)
        if evalmode:
            self.model.train()
        res = unravel_dict(res)

        for callback in self.model_callbacks:
            callback(self.model, self.optimizer, res, epoch, step)

        return ins, pred, tgs, res

    def predict(self, ins, freeze=False):
        """ Executes a forward pass on the model

        :param ins: The input to use
        :type ins: Tensor
        :param freeze: Whether network should be frozen
        :type freeze: bool, optional
        :return: The prediction
        :rtype: Tensor
        """
        if freeze:
            backup = utils.freeze(self.model)
        pred = self.model(ins)
        if freeze:
            utils.unfreeze(self.model, backup)

        return pred

    def summary(self, input_shape):
        """ Prints a Keras-like summary of the model.
        Code inspired from `this PyTorch issue <https://github.com/pytorch/pytorch/issues/2001>`.

        :param input_shape: The dimensions of the input that will be given to the network (without the batch size)
        :type ins: iterable
        """
        print()
        print('                       Strategy Summary')
        print('----------------------------------------------------------------')
        optim_name, optim_params = utils.get_optimizer_params(self.optimizer)
        print('Optimizer: {} with parameters:'.format(optim_name))
        for key, val in optim_params.items():
            print('\t{}:'.format(key), val)
        print('Loss: {}'.format(self.loss))
        device = 'CPU' if not self.cuda else 'GPU'
        model = self.model
        if self.multi_gpu:
            device = '{} {}'.format(torch.cuda.device_count(), device)
            model = self.model.module
        mod_type = model.__class__.__name__
        print('Running on {}.'.format(device))
        print('Model type is', mod_type)
        utils.summary(model, input_shape)

    def get_metadata(self):
        return self.metadata
