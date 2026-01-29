# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 12:53:39 2025

@author: boos
"""

import fibretracker as ft
import fibretracker_helper as fth
import helper as h
import tifffile as tf
import os, time
from datetime import datetime


##################### Input ######################

print("Select a 3D .tif to be analyzed.")
data_path = h.select_file_dir.select_file()

read_tracks = False

voxel_size = 0.8

load_inp = False

geodict_export = True

manipulate_tracks = False
 
###################################################


# Get the current date and time
execution_time = datetime.now()

exec_time = execution_time.strftime('%Y-%m-%d_%H_%M_%S')

t0 = time.time()
    
if data_path == r"":
    print("No volume selected.")
    
else:    
    V_UD = tf.imread(data_path)
    V_UD_name = os.path.splitext(os.path.basename(data_path))[0]
    result_folder = os.path.join(os.path.dirname(data_path), "result", f"{exec_time}_{V_UD_name}")
    h.ensure_dir_exists.ensure_directory_exists(result_folder)
    
    settings_folder = os.path.join(result_folder, "settings")
    h.ensure_dir_exists.ensure_directory_exists(settings_folder)
    
    if read_tracks:
        print("Select tracks.pkl")
        tracks = h.pkl.load_pkl(h.select_file_dir.select_file())
        
        if manipulate_tracks:
            log = fth.json_logger(os.path.join(settings_folder,"manipulate_tracks.json"))
            filter_small_tracks_start_end_logged = fth.capture_args_with_timing(log, exclude=("tracks",))(fth.filter_small_tracks_start_end)
            
            tracks = filter_small_tracks_start_end_logged(
                tracks = tracks,
                vol_depth = V_UD.shape[0],
                min_length = 150,
                z_crop = 150,
                voxel_size = 1.0)
    else:
        
        ############ Fiber Center Detection ###############
        
        log = fth.json_logger(os.path.join(settings_folder,"coords_detection.json"))
        get_fibre_coords2_logged = fth.capture_args_with_timing(log, exclude=("vol",))(ft.models.get_fibre_coords2)
        
        coords = get_fibre_coords2_logged(V_UD, 
                                             method     = "log", 
                                             min_sigma  = 3,
                                             max_sigma  = 4, 
                                             num_sigma  = 2, 
                                             threshold  = 0.05,
                                             overlap    = 0.5)
        
        ############ Fiber Tracking ###############
        
        log = fth.json_logger(os.path.join(settings_folder,"fiber_tracking.json"))
        track_fibres_logged = fth.capture_args_with_timing(log, exclude=("coords",))(ft.models.track_fibres)
        
        tracks = track_fibres_logged(coords=coords, 
                                             max_skip   = 2,
                                             track_min_length   = 25,
                                             momentum   = 0.8,
                                             smoothtrack_gaussian = False)
        
   
        h.pkl.save_pkl(os.path.join(result_folder, "found_tracks.pkl"), tracks)
            
    # Compute lengths for all tracks
    track_lengths = [fth.track_length(t, voxel_size) for t in tracks]
    
    ############ Optional: Visualization ###############
    
    ft.viz.plot_tracks(tracks, V_UD.shape, result_folder, show=False)
   
    log = fth.json_logger(os.path.join(settings_folder,"single_track_overlay.json"))
    overlay_tracks_one_slice_logged = fth.capture_args_with_timing(log, exclude=("tracks","vol",))(fth.overlay_tracks_one_slice)
    
    overlay_tracks_one_slice_logged(
        tracks, 
        vol = V_UD, 
        result_folder = result_folder,
        slice_id = 69, 
        radius = 2.0,
        colors=[(255,0,0)])
    
    log = fth.json_logger(os.path.join(settings_folder,"overlay_single_alpha_slice.json"))
    overlay_single_alpha_slice_logged = fth.capture_args_with_timing(log, exclude=())(fth.qual_analysis.overlay_single_alpha_slice)
    
    overlay_single_alpha_slice_logged(
            result_folder = result_folder,
            settings_folder = settings_folder,
            slice_id = 125,
            alpha_list = [8.0, 9.0, 10.0],
            make_overlay = True)
    
    ############ Fiber Length Location ###############
    
    log = fth.json_logger(os.path.join(settings_folder,"length_vs_slice_heatmap.json"))
    heatmap_length_vs_slice_nbins_logged = fth.capture_args_with_timing(log, exclude=("tracks",))(fth.qual_analysis.heatmap_length_vs_slice_nbins)
    
    H, bin_edges = heatmap_length_vs_slice_nbins_logged(
            tracks,
            nz=V_UD.shape[0],
            track_format="xyz",
            length_mode="z_span",
            voxel_size_um=voxel_size,
            n_length_bins=10,                 
            counting="per_slice_presence",
            normalize_per_slice = "mid_slice_only",
            title = "Track-length frequency vs slice",
            save_png=os.path.join(result_folder, "length_vs_slice_heatmap.png"),
            show=False
            
        )


    ############ FOD 1D ###############
    
    log = fth.json_logger(os.path.join(settings_folder,"FOD_angles_calculation.json"))
    compute_signed_orientations_logged = fth.capture_args_with_timing(log, exclude=("tracks",))(fth.quant_analysis.compute_signed_orientations)
    
    angles = compute_signed_orientations_logged(
            tracks,
            axis=2,       # angle vs Z-axis
            sign_axis=0   # sign determined by X component
            )
    
    log = fth.json_logger(os.path.join(settings_folder,"FOD_plot.json"))
    plot_signed_FOD_logged = fth.capture_args_with_timing(log, exclude=("angles_list",))(fth.quant_analysis.plot_signed_FOD)
    
    plot_signed_FOD_logged([angles], result_folder, axis_name="Z")

    ############ FOD 2D ###############    
    
    # dirs = fth.quant_analysis.compute_fiber_directions_from_tracks(
    # tracks,
    # coord_order="xyz",
    # use_endpoints_only=True,
    # undirected=True
    # )
    
    # fth.quant_analysis.plot_xy_spherical_density(
    #     dirs,
    #     output_folder=result_folder,
    #     exec_time=exec_time,
    #     bins=120
    # )

    
    # ax_deg, ay_deg = fth.quant_analysis.compute_xy_tilt_angles_deg(
    # tracks,
    # coord_order="xyz",
    # use_endpoints_only=True,
    # undirected=True
    # )
    
    # fth.quant_analysis.plot_xy_tilt_density(
    #     ax_deg,
    #     ay_deg,
    #     output_folder=result_folder,
    #     exec_time=exec_time,
    #     lim_deg=2.5,   # adjust (e.g. 10, 20, 30)
    #     bins=36
    # )

    
    ############ CDF and FLD ###############
    
    log = fth.json_logger(os.path.join(settings_folder,"track_lengths_CDF_hist_plot.json"))
    plot_track_length_CDF_logged = fth.capture_args_with_timing(log, exclude=("x",))(fth.quant_analysis.plot_track_length_CDF)
    
    plot_track_length_CDF_logged(track_lengths, 
                           out_dir = result_folder,  
                           label = "tracks", 
                           unit = "µm", 
                           log_x = False, 
                           base_name = "track_lengths",
                           show = False,
                           plot_volume = True,
                           fiber_radius_um = 3.5)

    
    
    # ############ Ripley ###############
    
    # ############ Collective Motion ###############
    
    # ############ Alpha-shape ###############
    
    # log = fth.json_logger(os.path.join(settings_folder,"alpha_masks.json"))
    # create_alpha_shape_logged = fth.capture_args_with_timing(log, exclude=("tracks","vol",))(fth.quant_analysis.create_alpha_shape)
    
    # alpha_masks, alpha_labels = create_alpha_shape_logged(
    #                            tracks,
    #                            V_UD,
    #                            result_folder,
    #                            alpha = 8.0,
    #                            z_tolerance = 0.5,      # include track points with z within +/- this of slice index
    #                            subsample_step = 1,     # increase if too slow (2,3,5,...)
    #                            connectivity = 8,
    #                            select = False,
    #                            save_mask = True,
    #                            save_labels = False)
    
    ############ Fiber Overlap ###############
    
    log = fth.json_logger(os.path.join(settings_folder,"fiber_overlap_calc.json"))
    overlaps_hist_adjacent_knn_streaming_logged = fth.capture_args_with_timing(log, exclude=("tracks",))(fth.quant_analysis.overlaps_hist_adjacent_knn_streaming)
    
    res_adj = overlaps_hist_adjacent_knn_streaming_logged(
        tracks,
        voxel_size_um=voxel_size,
        bins_um= int(V_UD.shape[0]/30),
        max_um= V_UD.shape[0],
        k_neighbors=8,
    )
    
    log = fth.json_logger(os.path.join(settings_folder,"fiber_overlap_plot.json"))
    plot_overlaps_hist_streaming_with_lognorm_logged = fth.capture_args_with_timing(log, exclude=("res",))(fth.quant_analysis.plot_overlaps_hist_streaming_with_lognorm)
    
    plot_overlaps_hist_streaming_with_lognorm_logged(
        res_adj,
        title="Adjacent fibre overlap (log-normal fit)",
        result_folder=result_folder,
        as_density=True,
    )
    
    ############ Log General Metadata ###############
    
    run_params = {
        "data_path": str(data_path),
        "geodict_export": geodict_export,
        "voxel_size_um": voxel_size,
        "read_tracks": read_tracks,
        "manipulate_tracks": manipulate_tracks,
        "timestamp_iso": datetime.now().isoformat(),
        "duration" : t0 - time.time()
    }



    
    ############ Compare Grountruth and Found Tracks ###############

if load_inp:
    print("Select a .inp file for the comparison with 3D .tif (optional).")
    gt_path = h.select_file_dir.select_file()
    
    fiber_stats, tracks_gt = fth.quant_analysis.get_fiber_stats(gt_path, False, False, "µm")
    
    tracks_gt = fth.create_ordered_tracks(tracks_gt, voxel_size, geodict_export)
    
    h.pkl.save_pkl(os.path.join(result_folder, "found_tracks_gt.pkl"), tracks_gt)
    
    track_lengths_gt = [row[2] for row in fiber_stats]  # total_length column
    # real = tf.imread(r"D:\rCF_paper\artificial_vol\straight_only\straight_only.tif")
    
    
    # Just make plots on screen (no files)
    # res = fth.quant_analysis.compare_length_sets(track_lengths, track_lengths_gt, label_a="Found Fibers", label_b="Ground Truth", unit="µm", show=True)
    # fth.quant_analysis.print_compare_report(res, "Found Fibers", "Ground Truth")
    
    # Or save PNGs to a folder
    payload = fth.quant_analysis.compare_and_save(track_lengths, track_lengths_gt, out_dir=result_folder, 
                                                  label_a="Found Fibers", label_b="Ground Truth", 
                                                  unit="µm", base_name="Found vs Ground Truth", log_x=False, show=False)
    
    test = fth.qual_analysis.paint_fibers_from_tracks(V_UD.shape, tracks, 4.5)#, (1.0,1.0,1.0))
    
    
    iou = fth.quant_analysis.iou(V_UD, test)
    
    dice = fth.quant_analysis.dice(V_UD, test)
    
    diff_img = fth.qual_analysis.make_diff_image(V_UD, test, os.path.join(result_folder, "diff_img.tif"))
    
    fth.quant_analysis.export_results(None, iou, dice, result_folder)

else:
    gt_path = ""

run_params["gt_path"] = str(gt_path)
h.json.save_json(run_params, os.path.join(settings_folder, "general_settings.json"))