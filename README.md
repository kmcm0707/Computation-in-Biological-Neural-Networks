# JAX inner loop

Run from the repository root:

```powershell
python code/main.py
```

The runner loads `results_4/Jax_13_chem_DSEF_full_sweep_200/1/400/meta_learner_model.eqx`, uses EMNIST images in `data/emnist`, and writes run output under `results_4/inner_loop/70`.

Install the Python dependencies first (for example, in a virtual environment):

```powershell
python -m pip install -r requirements.txt
```

Then run from the repository root with `python code/main.py`. The model and EMNIST data paths are resolved relative to the repository, so the command also works when launched from another working directory.

To save the sampled complex and backpropagation weight updates needed by
`code/alignment_angle_bp.py`, enable the diagnostics while running training:

```powershell
$env:CBNN_SAVE_WEIGHT_DIAGNOSTICS = "1"
python code/main.py
```

This writes `weight_changes_episode_XXXX.npy` and `bp_weight_changes_episode_XXXX.npy`
plus layer-shape metadata under `results_4/inner_loop/70`. These snapshot files
are large. Then run `python code/alignment_angle_bp.py` to create the alignment
plot for the episode and layer selected near the top of that script.

