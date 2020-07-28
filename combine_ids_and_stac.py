import numpy as np
import view_stac
from scipy.io import loadmat, savemat
import pickle
import os

M = loadmat('inds.mat')
inds = M['inds_total'][:]

# stac_files = ["/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/results/long_clips/Rat7M/JDM31/Recording_calibrated_hires_5/total.p",
# 			  "/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/results/long_clips/Rat7M/JDM48/Recording_day5_caff_syncedvid_5/total.p",
# 			  "/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/results/long_clips/Rat7M/JDM48/Recording_day7_cafftwo_nopedestal_5/total.p",
# 			  "/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/results/long_clips/Rat7M/JDM51/Recording_JDM51_fullmarkers_5/total.p",
# 			  "/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/results/long_clips/Rat7M/JDM52/Recording_caff_esu2prime_recording_5/total.p",
# 			  "/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/results/long_clips/Rat7M/JDM52/Recording_calib_caff_hires_5/total.p",
# 			  "/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/results/long_clips/Rat7M/JDM53/Recording_caff_esu2prime_recording_5/total.p"]

stac_files = ["Y:/Diego/tdata/dm/stac/results/long_clips/Rat7M/JDM31/Recording_calibrated_hires_5/total.p",
			  "Y:/Diego/tdata/dm/stac/results/long_clips/Rat7M/JDM48/Recording_day5_caff_syncedvid_5/total.p",
			  "Y:/Diego/tdata/dm/stac/results/long_clips/Rat7M/JDM48/Recording_day7_cafftwo_nopedestal_5/total.p",
			  "Y:/Diego/tdata/dm/stac/results/long_clips/Rat7M/JDM51/Recording_JDM51_fullmarkers_5/total.p",
			  "Y:/Diego/tdata/dm/stac/results/long_clips/Rat7M/JDM52/Recording_caff_esu2prime_recording_5/total.p",
			  "Y:/Diego/tdata/dm/stac/results/long_clips/Rat7M/JDM52/Recording_calib_caff_hires_5/total.p",
			  "Y:/Diego/tdata/dm/stac/results/long_clips/Rat7M/JDM53/Recording_caff_esu2prime_recording_5/total.p"]


for n_file, file in enumerate(stac_files):
	with open(file, 'rb') as f:
		in_dict = pickle.load(f)
	in_dict["inds"] = inds[n_file][0]
	in_dict["inds"] = [vec[0] for vec in in_dict["inds"]]		

	# Fix an indexing problem that occurs because of different subsampling methods. 
	if len(in_dict["inds"][0]) != in_dict['qpos'].shape[0]:
		in_dict["inds"] = [vec[:-1] for vec in in_dict["inds"]]

	print(len(in_dict["inds"]))
	print(len(in_dict["inds"][0]))
	print(in_dict['qpos'].shape)

	save_path = os.path.join(os.path.dirname(file), 'stac.p')
	with open(save_path, 'wb') as f:
		pickle.dump(in_dict, f, protocol=2)
	savemat(save_path.split('.')[0] + '.mat', in_dict)

	# Make a video for each snippet
	if not os.path.exists(os.path.join(os.path.dirname(file), 'videos')):
		os.makedirs(os.path.join(os.path.dirname(file), 'videos'))
	for n_snip, inds in enumerate(in_dict["inds"]):
		q = in_dict["qpos"][inds, :]
		kp_data = in_dict["kp_data"][inds, :]
		offsets = in_dict["offsets"]
		n_frames = np.sum(inds)
		save_path = os.path.join(os.path.dirname(file), 'videos', '%d.mp4' % (n_snip))
		view_stac.setup_visualization(param_path, q, offsets, kp_data, n_frames, render_video=True, save_path=None, headless=True)
	


