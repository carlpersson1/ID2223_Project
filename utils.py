import numpy as np
import pandas as pd


def encode_cycle(day_of_year, cycle_length, to_numpy=False):
    cos_encoding = np.cos(day_of_year * 2 * np.pi / cycle_length)
    sin_encoding = np.sin(day_of_year * 2 * np.pi / cycle_length)
    if not to_numpy:
        return cos_encoding, sin_encoding
    return cos_encoding.to_numpy(), sin_encoding.to_numpy()


if __name__ == '__main__':
    data = {'timestamp': [pd.Timestamp('2024-01-03 12:00'),
                          pd.Timestamp('2024-02-15 06:30'),
                          pd.Timestamp('2024-03-20 21:45')]}
    df = pd.DataFrame(data)
    df['day_cos'], df['day_sin'] = encode_cycle(df['timestamp'].dt.dayofyear, 365)
    df['hour_cos'], df['hour_sin'] = encode_cycle(df['timestamp'].dt.hour, 24)

    print(df.drop(columns='timestamp'))