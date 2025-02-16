import os

import torch
from metalearners.meta_learner_main import main
from metalearners.runner import runner_main
from nn.backprop import backprop_main

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    main()
    # runner_main()
    # backprop_main()
