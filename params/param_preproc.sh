data_path=/home/diego/data/dm/stac/Restpose_files/nolj_Recording_day8_caff1_nolj_imputed_JDM33.mat
param_path=./params/june3/JDM33.yaml
start_frame=108490
save_path=/home/diego/data/dm/stac/offsets/JDM33_june12.p

python optimize_scale_factor.py $data_path $param_path $start_frame
python compute_stac.py $data_path $param_path --save-path=$save_path --start-frame=$start_frame --process-snippet=False --verbose=True --visualize=True

# Add offset to the end of the params file
cat >> param_path << END
offset_path: $save_path
END
