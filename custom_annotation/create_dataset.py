import glob
import pickle
import pandas as pd

#f_dataset = open('./EPIC_detailed_action_labels.pkl'.format(input_name), 'w', encoding='UTF-8')

cols = ['mode','participant_id','video_id','narration','start_timestamp','stop_timestamp','start_frame','stop_frame','verb','verb_class','noun','noun_class','fps']
df = pd.DataFrame(index=[], columns=cols)
files = glob.glob("./annotations/processed/*.txt")

for file in files:
    f_anno = open(file, 'r', encoding='UTF-8')
    for line in f_anno:
        line = line.replace('\n', '')
        line = line.split(', ')
        record = pd.Series([line[0],line[1],line[2],'{} {}'.format(line[3],line[4]),line[5],line[6],line[7],line[8],line[9],line[10],line[11],line[12],line[13].replace('fps:','')], index=df.columns)
        df = df.append(record, ignore_index=True)
    f_anno.close()

print(df)
f = open('./annotations/EPIC_custom_action_labels.pkl', 'wb')
pickle.dump(df, f)
f.close
