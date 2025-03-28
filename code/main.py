import os

import torch
from metalearners.meta_learner_main import main
from metalearners.rnn_meta_learner_main import main_rnn
from metalearners.runner import runner_main
from misc.load_from_params import load_model
from nn.backprop import backprop_main
from nn.rnn_backprop import rnn_backprop_main

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    # main()
    # runner_main()
    # load_model()
    # backprop_main()
    # rnn_backprop_main()
    main_rnn()
