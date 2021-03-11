import copy
import gzip
import os
import pickle
import struct
import types

import cloudpickle
# from bt_rts.thrift.gen.scoring import TScoringRequest, TScoringResponse
# from bt_rts.scoring.recommenders.base import AbstractRecommender

import logging
LOG = logging.getLogger(__name__)


class ScopedCloudPickler(cloudpickle.CloudPickler):
    dispatch = cloudpickle.CloudPickler.dispatch.copy()

    def __init__(self, *args, mask_modules=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_modules = tuple(mask_modules)

    def save_global(self, obj, name=None, pack=struct.pack):
        modname = getattr(obj, "__module__", None)
        if modname is None:
            modname = pickle.whichmodule(obj, name)

        if any(modname.startswith(m) for m in self.mask_modules):
            obj = copy.copy(obj)
            obj.__module__ = '__main__'

        return super().save_global(obj, name, pack)

    dispatch[type] = save_global
    dispatch[types.ClassType] = save_global

    def save_function(self, obj, name=None):
        if name is None:
            name = obj.__name__
        modname = pickle.whichmodule(obj, name)

        if any(modname.startswith(m) for m in self.mask_modules):
            obj = copy.copy(obj)
            obj.__module__ = '__main__'

        return super().save_function(obj, name=name)

    dispatch[types.FunctionType] = save_function


# class CloudPickleRecommender(AbstractRecommender):
#     def __init__(self, model, mask_modules='bt_ai'):
#         if not mask_modules:
#             mask_modules = ()
#         if isinstance(mask_modules, str):
#             mask_modules = (mask_modules,)
#         else:
#             mask_modules = tuple(mask_modules)
#
#         self.model = model
#         self.mask_modules = mask_modules
#
#     def __call__(self, request: TScoringRequest, additional_data):
#         def convert_key(k):
#             if isinstance(k, str):
#                 return k.encode('UTF-8')
#             return k
#
#         result = self.model.predict(request, additional_data)
#         return TScoringResponse(
#             scored_candidates={
#                 convert_key(k): float(v)
#                 for k, v in result.items() if v is not None
#             }
#         )
#
#     def write(self, data_dir: str):
#         model_path = os.path.join(data_dir, 'model.pkl')
#         with open(model_path, 'wb') as fp:
#             pickler = ScopedCloudPickler(fp, pickle.HIGHEST_PROTOCOL, mask_modules=self.mask_modules)
#             pickler.dump(self.model)
#
#         print('serialized model with dependencies on %s', {m.__name__ for m in pickler.modules})
#
#     @classmethod
#     def read(cls, data_dir: str):
#         model_path = os.path.join(data_dir, 'model.pkl')
#         with open(model_path, 'rb') as fp:
#             model = cloudpickle.load(fp)
#         return cls(model)


class CloudPickleBatchScorer:
    def __init__(self, model, mask_modules=('bt_ai',)):
        self.model = model
        self.mask_modules = mask_modules

    def dump(self, filename):
        with gzip.open(filename, 'wb') as fp:
            pickler = ScopedCloudPickler(fp, pickle.HIGHEST_PROTOCOL, mask_modules=self.mask_modules)
            pickler.dump(self.model)

    @classmethod
    def load(cls, filename):
        with gzip.open(filename, 'rb') as fp:
            model = cloudpickle.load(fp)
        return cls(model)

    def score(self, user_data):
        return self.model.score(user_data)

    def scorer_hints(self):
        return self.model.scorer_hints()
