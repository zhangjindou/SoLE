import numpy as np
from hyperopt import hp

# required params:
# - embedding_size
# - lr
# - batch_size
# - max_iter
# - neg_ratio
# - contiguous_sampling
# - valid_every: set it to 0 to enable early stopping

param_space_Complex = {
    # "embedding_size": hp.quniform("embedding_size", 50, 200, 10),
    "embedding_size": 200,
    "l2_reg_lambda": hp.qloguniform("l2_reg_lambda", np.log(1e-3), np.log(1e-1), 1e-3),
    "lr": hp.qloguniform("lr", np.log(1e-4), np.log(1e-2), 1e-4),
    "batch_size": 5000,
    "max_iter": 100000,
    "neg_ratio": 1,
    "contiguous_sampling": False,
    "valid_every": 5000,
}

param_space_Complex_fb15k = {
    "embedding_size": 200,
    "l2_reg_lambda": 0.01,
    "lr": 0.001,
    "batch_num": 100,
    "max_iter": 100000,
    "neg_ratio": 10,
    "contiguous_sampling": False,
    "valid_every": 10000,
}

param_space_SoLE_Complex_fb15k = {
    "embedding_size": 300,
    "l2_reg_lambda": 0.01,
    "lr": 0.001,
    "batch_num": 100,
    "NNE_enable": True,
    "max_iter": 100000,
    "neg_ratio": 6, # 2,6,10
    "contiguous_sampling": False,
    "valid_every": 10000,
}


param_space_Complex_db100k = {
    "embedding_size": 150,
    "l2_reg_lambda": 0.03,
    "lr": 0.001,
    "batch_num": 100,
    "max_iter": 100000,
    "neg_ratio": 10,
    "contiguous_sampling": False,
    "valid_every": 10000,
}

param_space_SoLE_Complex_db100k = {
    "embedding_size": 300,
    "l2_reg_lambda": 0.03,
    "lr": 0.001,
    "batch_num": 100,
    "NNE_enable": False,
    "max_iter": 100000,
    "neg_ratio": 6, # 2,6,10
    "contiguous_sampling": False,
    "valid_every": 10000,
}


param_space_dict = {

    "Complex_fb15k": param_space_Complex_fb15k,
    "SoLE_Complex_fb15k": param_space_SoLE_Complex_fb15k,
    "Complex_db100k": param_space_Complex_db100k,
    "SoLE_Complex_db100k": param_space_SoLE_Complex_db100k,
}

int_params = [
    "embedding_size", "batch_size", "max_iter", "neg_ratio", "valid_every", "k",
    "fe_size", "hidden_size", "hidden_layers",
]

class ModelParamSpace:
    def __init__(self, learner_name):
        s = "Invalid model name! (Check model_param_space.py)"
        assert learner_name in param_space_dict, s
        self.learner_name = learner_name

    def _build_space(self):
        return param_space_dict[self.learner_name]

    def _convert_into_param(self, param_dict):
        if isinstance(param_dict, dict):
            for k, v in param_dict.items():
                if k in int_params:
                    param_dict[k] = int(v)
                elif isinstance(v, list) or isinstance(v, tuple):
                    for i in range(len(v)):
                        self._convert_into_param(v[i])
                elif isinstance(v, dict):
                    self._convert_into_param(v)
        return param_dict
