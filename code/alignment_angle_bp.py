"""Compare one sampled complex update with its backprop update."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import math


RESULTS = Path(__file__).resolve().parents[1] / "results_4" / "inner_loop" / "70"
EPISODE = 1
LAYER = 4



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


def main() -> None:
    prefix = f"episode_{EPISODE:04d}"
    cs_path = RESULTS / f"weight_changes_{prefix}.npy"
    bp_path = RESULTS / f"bp_weight_changes_{prefix}.npy"
    cs = np.load(cs_path)
    bp = np.load(bp_path)
    cs_shapes = np.load(RESULTS / f"weight_changes_{prefix}_layer_shapes.npy")
    bp_shapes = np.load(RESULTS / f"bp_weight_changes_{prefix}_layer_shapes.npy")

    if cs.ndim != 4 or bp.ndim != 4:
        raise ValueError(f"Expected padded 4D arrays; got complex={cs.shape}, BP={bp.shape}")
    if cs.shape != bp.shape:
        raise ValueError(f"Saved array shapes differ: complex={cs.shape}, BP={bp.shape}")
    if not 0 <= LAYER < cs.shape[0]:
        raise ValueError(f"Layer index {LAYER} is outside the saved layer range 0..{cs.shape[0] - 1}")
    if cs_shapes[LAYER].tolist() != bp_shapes[LAYER].tolist():
        raise ValueError(
            f"Layer {LAYER} shapes differ: complex={cs_shapes[LAYER]}, BP={bp_shapes[LAYER]}"
        )


    index_list=[]
    angle_list=[]
    # Infer the interval from the compact saved history so this works with
    # both older 20-image samples and new 3-image samples.
    num_training_images = 350
    sample_count = cs.shape[1]
    if sample_count == 0:
        raise ValueError("No saved weight-update samples were found")
    sample_interval = (num_training_images + sample_count - 1) // sample_count
    sample_indices = range(0, num_training_images, sample_interval)
    if len(sample_indices) != sample_count:
        raise ValueError(
            f"Cannot map {sample_count} saved samples to {num_training_images} training images"
        )

    for slot, image_index in enumerate(sample_indices):

        rows, cols = map(int, cs_shapes[LAYER])
        cs_update = cs[LAYER, slot, :rows, :cols]
        bp_update = bp[LAYER, slot, :rows, :cols]
        cosine = compute_cosine_similarity(cs_update, bp_update)
        angle = math.degrees(math.acos(cosine))
        angle_list.append(angle)
        index_list.append(image_index)
        print(
            f"Episode {EPISODE}, layer {LAYER}, training image {image_index + 1} (zero-based index {image_index}): "
            f"angle={angle:.4f} degrees, cosine={cosine:.8f} "
            f"(complex norm={np.linalg.norm(cs_update):.6g}, "
            f"BP norm={np.linalg.norm(bp_update):.6g})"
        )

    plt.plot([index + 1 for index in index_list], angle_list, marker=".", markersize=3)
    plt.xlabel("Training image (one-based)")
    plt.ylabel("Alignment angle (degrees)")
    plt.title(f"BP alignment angle: episode {EPISODE}, layer {LAYER}")
    plt.grid(True, alpha=0.3)
    output_path = RESULTS / f"bp_alignment_episode_{EPISODE:04d}_layer_{LAYER}.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved alignment plot to {output_path}")


if __name__ == "__main__":
    main()
