import os
import subprocess
import ffmpeg
import pandas as pd


def convert_time_to_second(str1, str2):
    times1 = str1.split(':')
    times1[2] = times1[2].split('.')
    times2 = str2.split(':')
    times2[2] = times2[2].split('.')
    second1 = int(times1[0])*(60*60) + int(times1[1])*60 + int(times1[2][0])
    second2 = int(times2[0])*(60*60) + int(times2[1])*60 + int(times2[2][0])
    diff = second2 - second1
    return second1, second2, diff


# init
INPUT_DIR = 'symlink'
OUTPUT_DIR = 'symlink'
dataset = pd.read_pickle('../custom_annotation/annotations/EPIC_custom_action_labels_splited.pkl')
dataset = dataset.sort_values('video_id')
current_video = None
cap = None
outfile = open('./tmp/output.txt', 'w')


# clip
for d in dataset.itertuples():
    """
    d = Pandas(
        Index=198, mode='train', participant_id='P01', video_id='P01_01',
        narration='finely_chop courgette', start_timestamp='00:04:01.00', stop_timestamp='00:04:08.00',
        start_frame='14460', stop_frame='14880', verb='finely_chop', verb_class='8', noun='courgette', noun_class='69', fps='60'
    )
    """

    if not os.path.exists('./output_video/{}/{}'.format(OUTPUT_DIR, d.video_id)):
        os.mkdir('./output_video/{}/{}'.format(OUTPUT_DIR, d.video_id))

    person_id = d.video_id.split('_')[0].replace('P', '')
    video_path = './input_video/{}/{}/videos/{}.MP4'.format(INPUT_DIR, person_id, d.video_id)
    output_path = './output_video/{}/{}/{}_{}_{}.MP4'.format(OUTPUT_DIR, d.video_id, d.Index, d.verb, d.noun)

    if not os.path.exists(video_path):
        print('Not found video={}'.format(d.video_id))
        continue

    if os.path.exists(output_path):
        print('Already exist video={}, index={}'.format(d.video_id, d.Index))
        continue
    else:
        print('Processing video={}, index={}'.format(d.video_id, d.Index))

    start, _, diff = convert_time_to_second(d.start_timestamp, d.stop_timestamp)
    cmd = 'ffmpeg -ss {} -t {} -i {} -vf scale=480:-1 {}'.format(start, diff, video_path, output_path)
    subprocess.run(cmd, shell=True, stdout=outfile, stderr=outfile)


# finish
print('Completed processing.')
