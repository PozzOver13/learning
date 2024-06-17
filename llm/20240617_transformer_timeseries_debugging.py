import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    names = ['year', 'month', 'day', 'dec_year', 'sn_value',
             'sn_error', 'obs_num', 'unused1']
    df = pd.read_csv("https://data.heatonresearch.com/data/t81-558/SN_d_tot_V2.0.csv",
                     sep=';', header=None, names=names,
                     na_values=['-1'], index_col=False)

    # Data Preprocessing
    start_id = max(df[df['obs_num'] == 0].index.tolist()) + 1
    df = df[start_id:].copy()
    df['sn_value'] = df['sn_value'].astype(float)
    df_train = df[df['year'] < 2000]
    df_test = df[df['year'] >= 2000]

    spots_train = df_train['sn_value'].to_numpy().reshape(-1, 1)
    spots_test = df_test['sn_value'].to_numpy().reshape(-1, 1)

    scaler = StandardScaler()
    spots_train = scaler.fit_transform(spots_train).flatten().tolist()
    spots_test = scaler.transform(spots_test).flatten().tolist()

    # Sequence Data Preparation
    SEQUENCE_SIZE = 10

    def to_sequences(seq_size, obs):
        x = []
        y = []
        for i in range(len(obs) - seq_size):
            window = obs[i:(i + seq_size)]
            after_window = obs[i + seq_size]
            x.append(window)
            y.append(after_window)
        return torch.tensor(x, dtype=torch.float32).view(-1, seq_size, 1), torch.tensor(y, dtype=torch.float32).view(-1,
                                                                                                                     1)


    x_train, y_train = to_sequences(SEQUENCE_SIZE, spots_train)
    x_test, y_test = to_sequences(SEQUENCE_SIZE, spots_test)
