import pandas as pd

#verb = 'cut'
#noun = 'kiwi'
mode = 'train'
video_name = 'P01_01'
# cheese, fish, cucumber, banana
# squash, mushroom, kiwi

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    df1 = pd.read_pickle('../train_val/EPIC_train_action_labels.pkl')
    df2 = pd.read_pickle('../train_val/EPIC_val_action_labels.pkl')
    df = pd.concat([df1,df2])

    print(df.columns)

    df = df[df['video_id'] == video_name]

    text_file = open("./test.txt", "wt")
    for d in df.itertuples():
        d = list(d)
        del d[12:14]
        text_file.write("{}\n".format(', '.join([str(i) for i in d])))
    text_file.close()
