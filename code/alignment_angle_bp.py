"""Compare sampled complex updates with backprop updates across episodes and layers."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import math


RESULTS = Path(__file__).resolve().parents[1] / "results_4" / "inner_loop" / "200"
EPISODES = [8]      # Episode files are numbered starting at 1
LAYERS = [0, 1, 2, 3,4]     # Edit with the layers you want to run (or set to None for all layers)


def compute_cosine_similarity(delta_w_cs: np.ndarray, delta_w_bp: np.ndarray) -> float:
    """Return cosine similarity, or raise when either update is zero."""
    u = np.asarray(delta_w_cs, dtype=np.float64).ravel()
    v = np.asarray(delta_w_bp, dtype=np.float64).ravel()
    if u.shape != v.shape:
        raise ValueError(f"Update shapes differ: complex={u.shape}, BP={v.shape}")

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        raise ValueError(
            f"Cosine is undefined for a zero update (complex norm={norm_u:g}, BP norm={norm_v:g}). "
            "Regenerate the saved arrays after fixing compute_bp_delta_w."
        )

    return float(np.clip(np.dot(u, v) / (norm_u * norm_v), -1.0, 1.0))


def process_episode(episode: int, layers: list[int] | None = None) -> None:
    prefix = f"episode_{episode:04d}"
    cs = np.load(RESULTS / f"weight_changes_{prefix}.npy")
    bp = np.load(RESULTS / f"bp_weight_changes_{prefix}.npy")
    cs_shapes = np.load(RESULTS / f"weight_changes_{prefix}_layer_shapes.npy")
    bp_shapes = np.load(RESULTS / f"bp_weight_changes_{prefix}_layer_shapes.npy")

    if cs.ndim != 4 or bp.ndim != 4:
        raise ValueError(f"Expected padded 4D arrays; got complex={cs.shape}, BP={bp.shape}")
    if cs.shape != bp.shape:
        raise ValueError(f"Saved array shapes differ: complex={cs.shape}, BP={bp.shape}")

    num_training_images = 200*5
    sample_count = cs.shape[1]
    if sample_count == 0:
        raise ValueError("No saved weight-update samples were found")
    sample_interval = (num_training_images + sample_count - 1) // sample_count
    sample_indices = range(0, num_training_images, sample_interval)
    if len(sample_indices) != sample_count:
        raise ValueError(
            f"Cannot map {sample_count} saved samples to {num_training_images} training images"
        )

    target_layers = range(cs.shape[0]) if layers is None else layers

    for layer in target_layers:
        if not 0 <= layer < cs.shape[0]:
            raise ValueError(f"Layer index {layer} is outside the saved layer range 0..{cs.shape[0] - 1}")
        if cs_shapes[layer].tolist() != bp_shapes[layer].tolist():
            raise ValueError(
                f"Layer {layer} shapes differ: complex={cs_shapes[layer]}, BP={bp_shapes[layer]}"
            )

        index_list = []
        angle_list = []
        cosine_list = []
        rows, cols = map(int, cs_shapes[layer])

        for slot, image_index in enumerate(sample_indices):
            cs_update = cs[layer, slot, :rows, :cols]
            bp_update = bp[layer, slot, :rows, :cols]
            cosine = compute_cosine_similarity(cs_update, bp_update)
            angle = math.degrees(math.acos(cosine))
            angle_list.append(angle)
            index_list.append(image_index)
            cosine_list.append(cosine)
            print(
                f"Episode {episode}, layer {layer}, training image {image_index + 1} (zero-based index {image_index}): "
                f"angle={angle:.4f} degrees, cosine={cosine:.8f} "
                f"(complex norm={np.linalg.norm(cs_update):.6g}, "
                f"BP norm={np.linalg.norm(bp_update):.6g})"
            )

        # 1. Line plot
        plt.plot([index + 1 for index in index_list], cosine_list, marker=".", markersize=3)
        plt.xlabel("Training image (one-based)")
        plt.ylabel("Cosine of Alignment angle")
        plt.title(f"BP alignment angle: episode {episode}, layer {layer}")
        plt.grid(True, alpha=0.3)
        output_path = RESULTS / f"bp_alignment_episode_{episode:04d}_layer_{layer}.png"
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved alignment plot to {output_path}")

        # 2. Symmetric histogram
        limit = max(abs(min(cosine_list)), abs(max(cosine_list)))
        limit = min(1.0, limit * 1.05) if limit > 0 else 1.0
        plt.hist(cosine_list, bins=20, range=(-limit, limit), density=True, edgecolor="black")
        plt.xlim(-limit, limit)
        plt.axvline(0, color="red", linestyle="--", linewidth=1)
        plt.xlabel("Cosine of alignment angle")
        plt.ylabel("Density")
        plt.title(f"BP alignment angle: episode {episode}, layer {layer}")
        output_path = RESULTS / f"bp_alignment_episode_{episode:04d}_layer_{layer}_hist.png"
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()


def main() -> None:
    for episode in EPISODES:
        process_episode(episode, layers=LAYERS)


if __name__ == "__main__":
    main()
