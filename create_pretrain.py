import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

labels = ['S', 'R', 'G', 'M', 'P', 'D', 'N']
data = []

for raga in os.listdir('pitch'):
    for file in os.listdir(os.path.join('pitch', raga)):
        pitch_df = pd.read_csv(os.path.join('pitch', raga, file), sep='\t', header=None, names=['time', 'pitch'])
        annotations_df = pd.read_csv(os.path.join('annotations', raga, file.replace('.tsv', '_phonemes.tsv')), sep='\t')

        times = pitch_df['time'].values
        pitch = pitch_df['pitch'].values

        for i, row in annotations_df.iterrows():
            if i == 0 or i == len(annotations_df) - 1:
                continue

            start_time = float(row['Begin time'].split(':')[-1])
            end_time = float(row['End time'].split(':')[-1])

            annotation = labels.index(row['Annotation'][0])

            prec_df = pitch_df[(pitch_df['time'] >= (start_time - 0.5)) & (pitch_df['time'] <= start_time)]
            curr_df = pitch_df[(pitch_df['time'] >= start_time) & (pitch_df['time'] <= end_time)]
            succ_df = pitch_df[(pitch_df['time'] >= end_time) & (pitch_df['time'] <= (end_time + 0.5))]
            if curr_df['pitch'].isna().all():
                continue
            if annotation * 100 < min(curr_df['pitch'].values) or annotation * 100 > max(curr_df['pitch'].values):
                continue

            data.append((
                annotation,
                -1,
                prec_df['pitch'].values,
                curr_df['pitch'].values,
                succ_df['pitch'].values
            ))

with open('dataset/pretrain.pkl', 'wb') as f:
    pickle.dump(data, f)
