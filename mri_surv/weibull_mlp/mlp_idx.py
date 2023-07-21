"""
module for testing/outputting RIDs used for each fold in Weibull models.
"""

import pandas as pd


FI_ADNI = 'metadata/data_processed/weibull_model_adni.csv'

df = pd.read_csv(
    FI_ADNI,
    usecols=['RID'] + [f'Fold{i}' for i in range(5)],
    dtype={'RID': str}
)

root = './results/exp_ids_weibull'

os.makedirs(root, exist_ok=True)

stage_map = {'train': 1, 'valid': 2, 'test': 3}

for key, val in stage_map.items():
    for fold in range(5):
        col = f'Fold{fold}'
        rid = df.loc[df[col] == val,'RID'].to_numpy()
        with open(os.path.join(root, f'ADNI_{key}_{fold}.csv'),'w') as fi:
            fi.write('rid\n')
            for x in rid: fi.write(str(x) + '\n') 