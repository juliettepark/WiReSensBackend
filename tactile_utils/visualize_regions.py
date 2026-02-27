"""
Visualize right_hand_regions: how (row, col) indices map to hand region names.
Run: python utils/visualize_regions.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib

matplotlib.use("Agg")

# Must match utils/tactile_handpose_utils.py
right_hand_regions = {
    "t2": (slice(13, 16), slice(12, 16)),
    "t1": (slice(11, 13), slice(12, 16)),
    "pa1": (slice(0, 6), slice(12, 16)),
    "pa2": (slice(6, 11), slice(12, 16)),
    "pa3": (slice(6, 11), slice(9, 12)),
    "pa4": (slice(0, 6), slice(9, 12)),
    "i1": (slice(9, 11), slice(6, 9)),
    "i2": (slice(9, 11), slice(3, 6)),
    "i3": (slice(9, 11), slice(0, 3)),
    "m1": (slice(6, 8), slice(6, 9)),
    "m2": (slice(6, 8), slice(3, 6)),
    "m3": (slice(6, 8), slice(0, 3)),
    "r1": (slice(3, 5), slice(6, 9)),
    "r2": (slice(3, 5), slice(3, 6)),
    "r3": (slice(3, 5), slice(0, 3)),
    "p1": (slice(0, 2), slice(6, 9)),
    "p2": (slice(0, 2), slice(3, 6)),
    "p3": (slice(0, 2), slice(0, 3)),
}


def build_region_grid():
    """Build 16x16 array: grid[row, col] = region index (0..n_regions-1)."""
    grid = np.full((16, 16), -1, dtype=int)
    name_to_id = {name: i for i, name in enumerate(right_hand_regions)}
    for name, (row_sl, col_sl) in right_hand_regions.items():
        rid = name_to_id[name]
        # grid[row_sl, col_sl] = rid
        grid[col_sl, row_sl] = rid
    return grid, list(right_hand_regions.keys())


def main():
    grid, names = build_region_grid()
    n_regions = len(names)

    # Colors: one per region; index 0 = uncovered (gray)
    cmap = plt.cm.get_cmap("tab20", max(n_regions, 20))
    colors = [cmap(i) for i in range(n_regions)]
    colors_with_empty = [np.array([0.95, 0.95, 0.95, 1.0])] + colors
    cmap_custom = ListedColormap(colors_with_empty)
    # Plot grid+1 so that -1 -> 0 (gray), 0..n-1 -> 1..n (region colors)
    plot_grid = grid + 1
    plot_grid[grid == -1] = 0

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(plot_grid, cmap=cmap_custom, vmin=0, vmax=n_regions, aspect="equal")

    # Grid lines
    for i in range(17):
        ax.axhline(i - 0.5, color="black", linewidth=0.5)
        ax.axvline(i - 0.5, color="black", linewidth=0.5)
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(15.5, -0.5)

    # Label each cell with region name
    for r in range(16):
        for c in range(16):
            rid = grid[r, c]
            if rid >= 0:
                ax.text(c, r, names[rid], ha="center", va="center", fontsize=7, color="black")

    ax.set_xlabel("Column index (second slice in each region tuple)")
    ax.set_ylabel("Row index (first slice in each region tuple)")
    ax.set_title("right_hand_regions: 16×16 grid → hand regions\n(row, col) = (first slice, second slice)")

    # Legend
    legend_handles = [
        Patch(facecolor=colors[i], edgecolor="black", label=names[i]) for i in range(n_regions)
    ]
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

    plt.tight_layout()
    out_path = "hand_regions_grid_flipped.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()

    # Print index summary
    print("\nRegion → (row_slice, col_slice) (grid[row_slice, col_slice]):")
    for name, (row_sl, col_sl) in right_hand_regions.items():
        nr = row_sl.stop - row_sl.start
        nc = col_sl.stop - col_sl.start
        print(f"  {name}: rows {row_sl.start}:{row_sl.stop}, cols {col_sl.start}:{col_sl.stop}  ({nr}×{nc} cells)")


if __name__ == "__main__":
    main()
