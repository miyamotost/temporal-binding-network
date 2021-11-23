import pandas as pd

verb = 'cut'
noun = 'kiwi'
# cheese, fish, cucumber, banana
# squash, mushroom, kiwi

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    modes = ['train', 'val']
    for mode in modes:
        df = pd.read_pickle('../train_val/EPIC_{}_action_labels.pkl'.format(mode))
        df = df[(df['verb'] == verb) & (df['noun'] == noun)]

        #df = df[df['verb'] == verb]
        #print(df['noun'].value_counts())
        #continue

        text_file = open("./annotations/raw/output_{}_{}_{}.txt".format(mode, verb, noun), "wt")
        for d in df.itertuples():
            d = list(d)
            del d[12:14]
            text_file.write("{}\n".format(', '.join([str(i) for i in d])))
        text_file.close()
