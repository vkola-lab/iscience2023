import os
import pandas as pd
import numpy as np


def main():
    root = './results/exp_ids_'

    subdirs = ('cnn','weibull','mlp')

    fold = ('train','test','valid')

    dataset = 'ADNI'

    exp = list(range(5))

    for f in fold:
        for ex in exp:
            mods = []
            fi_list = []
            for model in subdirs:
                fi = root+model+f'/{dataset}_{f}_{ex}.csv'
                fi_list.append(fi)
                df = pd.read_csv(fi, dtype={'rid': str})
                df = np.sort(df.to_numpy(), axis=0)
                mods.append(df)
            mods = np.concatenate(mods,axis=1)
            print([f'{x},' for x in fi_list])
            assert(np.all([np.array_equal(mods[:,0],mods[:,i]) for i in range(mods.shape[1])]))

if __name__ == '__main__':
    main()