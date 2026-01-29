''' 
This module contains functions for plotting the tracks of fibres detected in the volume
'''
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_tracks(
        tracks: List[np.ndarray],
        shape: tuple,
        result_folder,
        grid: bool = False,
        show: bool = True
        ):
    
    '''Plot tracks of fibres detected in the volume

    Args:
        tracks: List of arrays of shape (n_points, 3)
        grid: Whether to show grid in the plot
    
    Returns:
        fig (matplotlib.figure.Figure): matplotlib figure object
    
    Example:
        ```python
        import fibretracker as ft

        # Load the volume and detected coordinates
        vol = ft.io.load("path/to/volume.txm")
        vol = ft.io.normalize(vol)
        vol = vol[100:350] # 250 slices along the z-axis
        detect_coords = ft.models.get_fibre_coords(vol)
        tracks_gauss = ft.models.track_fibres(coords=detect_coords, smoothtrack_gaussian=True)
        ft.viz.plot_tracks(tracks_gauss)
        ```

        ![viz tracks](figures/tracks_gauss.gif)


    '''

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    n_tracks = sum(t is not None and len(t) > 0 for t in tracks)

    for track in tracks:
        if track is None or len(track) == 0:
            continue
        ax.plot(track[:, 0], track[:, 1], track[:, 2])

    ax.grid(grid)
    ax.set_box_aspect((shape[2], shape[1], shape[0]))

    ax.text2D(
        0.02, 0.98,
        f"Found Tracks: {n_tracks}",
        transform=ax.transAxes,
        va="top",
        fontsize=20,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
    )


    plt.savefig(os.path.join(result_folder, "tracks_3D.png"), bbox_inches="tight", dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
