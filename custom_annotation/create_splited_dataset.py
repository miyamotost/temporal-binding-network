import glob
import random
import pandas as pd


def get_mode(num):
    if num>=1 and num<=65:
        return 'train'
    elif num>=66 and num<=80:
        return 'val'
    elif num>=81 and num<=100:
        return 'test'
    else:
        print('invalid num')
        exit()


df = pd.read_pickle('./annotations/EPIC_custom_action_labels.pkl')
for index, item in df.iterrows():
    mode = random.randint(1, 100)
    mode = get_mode(mode)
    df.loc[index,'mode'] = mode
df.to_pickle('./annotations/EPIC_custom_action_labels_splited.pkl')
