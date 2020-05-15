import os
import sys
import numpy as np
import pickle

# base_folder = "/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/results/long_clips/JDM31_day8"
base_folder = "/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/results/long_clips/Baseline1_JDM25/"
if not os.path.exists(base_folder):
    os.makedirs(base_folder)
# data_path = "/n/home02/daldarondo/LabDir/Jesse/Data/Dropbox_curated_sharefolders/mocap_data_diego/diego_mocap_files_rat_JDM31_day_8.mat"
data_path = "/n/home02/daldarondo/LabDir/Jesse/Data/Dropbox_curated_sharefolders/mocap_data_diego/Baseline1_JDM25.mat"
offset_path = "/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/offsets/july22/JDM25.p"
# param_path = "/n/home02/daldarondo/LabDir/Diego/code/dm/stac/params/july15/JDM31_DANNCE.yaml"
param_path = "/n/home02/daldarondo/LabDir/Diego/code/dm/stac/params/july15/JDM25.yaml"
temp_file_name = 'run_variables.p'

snippet_duration = 7200
clip_duration = 26460000
start_frames = np.arange(0, clip_duration, snippet_duration)
end_frames = start_frames + snippet_duration
# start_frames = start_frames[1000:1002]
# end_frames = end_frames[1000:10002]

def get_unfinished(base_folder, start_frames):
    is_unfinished = []
    for i, sf in enumerate(start_frames):
        if not os.path.exists(os.path.join(base_folder, '%d.p' % (sf))):
            is_unfinished.append(True)
        else:
            is_unfinished.append(False)
    return is_unfinished


def save_variables(start_frames, end_frames):
    out_dict = {'start_frames': start_frames, 'end_frames': end_frames}
    with open(temp_file_name, 'wb') as f:
        pickle.dump(out_dict, f)


def load_variables():
    with open(temp_file_name, 'rb') as f:
        in_dict = pickle.load(f)
    return in_dict['start_frames'], in_dict['end_frames']

# Print to std out if just asking for the number of commands
# (for sbatch --array=0-$print_num_commands ...)
if __name__ == "__main__":
    if len(sys.argv) >= 2:
        if sys.argv[1] == '--submit':
            save_variables(start_frames, end_frames)
            n_jobs = len(start_frames)
            print('Number of jobs: ', n_jobs)
            cmd = 'sbatch --array=0-%d --partition=shared,olveczky,serial_requeue --exclude=seasmicro25 cluster/submit_compute_stac_clip.sh' % (n_jobs - 1)
            print(cmd)
            os.system(cmd)

        if sys.argv[1] == '--submit-unfinished':
            is_unfinished = get_unfinished(base_folder, start_frames)
            start_frames = np.array([start_frames[i] for i, v in enumerate(is_unfinished) if v])
            end_frames = np.array([end_frames[i] for i, v in enumerate(is_unfinished) if v])
            # start_frames = start_frames[:1]
            # end_frames = end_frames[:1]
            save_variables(start_frames, end_frames)
            n_jobs = len(start_frames)
            print('Number of jobs: ', n_jobs)
            cmd = 'sbatch --array=0-%d --partition=shared,olveczky,serial_requeue --exclude=seasmicro25 cluster/submit_compute_stac_clip.sh' % (n_jobs - 1)
            print(cmd)
            # print(start_frames)
            # print(end_frames)
            os.system(cmd)

    # Otherwise, run the command for the appropriate job id
    else:
        import compute_stac
        task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        start_frames, end_frames = load_variables()
        start_frame = start_frames[task_id]
        end_frame = end_frames[task_id]
        save_path = os.path.join(base_folder, '%d.p' % (start_frame))
        compute_stac.handle_args(data_path,
                                 param_path,
                                 save_path=save_path,
                                 offset_path=offset_path,
                                 verbose=True,
                                 process_snippet=False,
                                 start_frame=start_frame,
                                 end_frame=end_frame,
                                 skip=1)
