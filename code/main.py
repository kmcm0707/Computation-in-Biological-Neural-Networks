import os

import torch
from metalearners.meta_learner_main import main  # noqa: F401
from metalearners.rnn_meta_learner_main import main_rnn  # noqa: F401
from metalearners.runner import runner_main  # noqa: F401
from misc.load_from_params import load_model  # noqa: F401
from nn.backprop import backprop_main  # noqa: F401
from nn.rnn_backprop import rnn_backprop_main  # noqa: F401

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    #main()
    # runner_main()
    # load_model()
    # backprop_main()
    # rnn_backprop_main()
    main_rnn()
