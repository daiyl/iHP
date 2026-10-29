import os
import torch
import importlib


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        modules = {
            'LSTM': 'LSTM', 'xLSTM': 'model_xlstm', 'xlstm': 'model_xlstm',
            'Mamba': 'Mamba', 'iTransformer': 'iTransformer', 'PatchTST': 'PatchTST',
            'DLinear': 'DLinear', 'TiDE': 'TiDE', 'SCINet': 'SCINet', 'TimesNet': 'TimesNet'
        }
        if args.model not in modules:
            raise ValueError(f"Unsupported model {args.model}; choose from {list(modules)}")
        self.model_dict = {args.model: importlib.import_module('models.' + modules[args.model])}

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
