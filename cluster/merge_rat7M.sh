#!/bin/bash
set -e
paths=("/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/results/long_clips/Rat7M/JDM31/Recording_calibrated_hires_7")
       # "/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/results/long_clips/Rat7M/JDM48/Recording_day5_caff_syncedvid_6" \
       # "/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/results/long_clips/Rat7M/JDM48/Recording_day7_cafftwo_nopedestal_6" \
       # "/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/results/long_clips/Rat7M/JDM51/Recording_JDM51_fullmarkers_6" \
       # "/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/results/long_clips/Rat7M/JDM52/Recording_caff_esu2prime_recording_6" \
       # "/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/results/long_clips/Rat7M/JDM52/Recording_calib_caff_hires_6" \
       # "/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/results/long_clips/Rat7M/JDM53/Recording_caff_esu2prime_recording_6")

echo ${#paths[@]}
for i in $(seq 0 $((${#paths[@]}-1)))
    do
        python merge_snippets.py ${paths[i]}
    done
