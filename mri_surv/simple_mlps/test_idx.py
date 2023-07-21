import pandas as pd
from simple_mlps.datas import ParcellationData

root = './results/exp_ids_mlp'

os.makedirs(root, exist_ok=True)

def test():
    for _exp in range(5):
        for stage in ('train', 'valid', 'test'): 
            pd = ParcellationData(_exp, stage=stage) 
            rid = pd.rid[pd.index_list]
            with open(os.path.join(root,f'ADNI_{stage}_{_exp}.csv'), 'w') as fi:
                fi.write('rid\n')
                for x in rid: fi.write(x + '\n')

test()