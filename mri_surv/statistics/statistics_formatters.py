from lifelines import KaplanMeierFitter, CoxPHFitter
from scipy import stats
from icecream import ic
from collections import defaultdict
from statsmodels.stats import multitest
from math import factorial
import pandas as pd
import numpy as np
import lifelines
import itertools

__all__ = ['KaplanMeierPairwise','ChiSquare', 'MannWhitneyU','KruskalWallis','PermutationTest','KaplanMeierStatistics',
            'permutation_test','difference_of_means','difference_of_medians','pairwise_mannwhitneyu',
            'pairwise_perm','chisq','compute_jaccard_similarity', 'kaplan_meier_estimator']

def comb(n,r):
    return factorial(n)/(factorial(r)*factorial(n-r))

class ChiSquare:
    def __init__(self, cross_tab):
        self.df = cross_tab
        self.columns = np.setdiff1d(list(cross_tab.columns),['All'])
        self.rows = np.setdiff1d(list(cross_tab.index),['All'])
        self.columns_name = cross_tab.columns.name
        assert(len(self.rows) == 2)
        self.omnibus_st = chisq(cross_tab.loc[self.rows, self.columns])
        self.omnibus_str = f'Omnibus for {self.columns_name}' + \
               _stringify_chisq(self.omnibus_st)
        proportions = self.df.loc[self.rows,self.columns].divide(
                    self.df.loc[self.rows, 'All'], axis='rows'
            )
        self.proportions = pd.melt(proportions.reset_index(),
                                  id_vars='Dataset',
                              value_name='Proportion').set_index([
                'Dataset', self.columns_name])
        self.chisq_pairwise()

    def chisq_pairwise(self):
        self.pairwise_stats = {}
        self.pairwise_str = ''
        all_values = self.df.loc[self.rows, 'All'].to_numpy().reshape((-1,1))
        if len(self.columns) < 3:
            return
        for col in self.columns:
            current_col = self.df.loc[self.rows, col].to_numpy().reshape((
                    -1,1))
            current_tbl = np.concatenate([current_col,
                                          all_values-current_col], axis=1)
            self.pairwise_stats[col] = chisq(current_tbl, nreps=len(
                    self.columns))
            self.pairwise_str += f'\n{self.columns_name},col {str(col)}: \n'\
                                     + _stringify_chisq(self.pairwise_stats[col])
            self.pairwise_str += '\tproportion ADNI vs NACC: {} vs {}\n'.format(
                    self.proportions.loc['ADNI',col].values[0],
                    self.proportions.loc['NACC', col].values[0]
            )
            self.pairwise_str += '\tcounts for ADNI vs NACC: {} vs {}\n'.format(
                    self.df.loc['ADNI','All'],
                    self.df.loc['NACC', 'All']
            )
    def __str__(self):
        return self.omnibus_str + '\n' + self.pairwise_str

class MannWhitneyU:
    def __init__(self, x, y, x_label, y_label, n_comparisons=1):
        self.x, self.y = x.reshape((-1,1)), y.reshape((-1,1))
        self.x_label, self.y_label = x_label, y_label
        self.n_comparisons = n_comparisons
        self.mannwhitneyu()

    def mannwhitneyu(self):
        x, y = self.x, self.y
        x_nan = sum(np.isnan(x))[0]
        n_x = sum(~np.isnan(x))[0]
        y_nan = sum(np.isnan(y))[0]
        n_y = sum(~np.isnan(y))[0]
        _, p_less = stats.mannwhitneyu(x[~np.isnan(x)],y[~np.isnan(y)],
                                       alternative='less')
        stat, p_greater = stats.mannwhitneyu(x[~np.isnan(x)],y[~np.isnan(y)], alternative='greater')
        if p_less*2*self.n_comparisons < 0.05:
            output_str = '<'
        elif p_greater*2*self.n_comparisons < 0.05:
            output_str = '>'
        else:
            output_str = '='
        output_str = f'{self.x_label}{output_str}{self.y_label}'
        self.stats = {
                'stat': stat,
                'p': min([min([p_less, p_greater])*2*self.n_comparisons,1]),
                'n_x': n_x,
                'n_y': n_y,
                'n_x_nan': x_nan,
                'n_y_nan': y_nan
        }
        self.string = output_str

    def __str__(self):
        str_list = [f'Wilcoxon Rank-Sum test: {self.string}\n'] + \
               [f'\t{key}={value}\n' for key,value in self.stats.items()]
        return ''.join(str_list)

class KruskalWallis:
    def __init__(self, x, label):
        self.x = x
        self.label = label
        self.kwtest()

    def kwtest(self):
        x = self.x
        n_x_nan = [np.sum(np.isnan(z)) for z in x]
        n_x = [np.sum(~np.isnan(z)) for z in x]
        st, p = stats.kruskal(*[z[~np.isnan(z)] for z in x],
                                       nan_policy='raise')
        self.stats = {
                'H': st,
                'df': len(x)-1,
                'p': p,
                'n_x': ','.join([str(y) for y in n_x]),
                'n_x_nan': ','.join([str(y) for y in n_x_nan]),
        }

    def __str__(self):
        str_list = [f'Kruskal-Wallis test: {self.label}\n'] + \
               [f'\t{key}={value}\n' for key,value in self.stats.items()]
        return ''.join(str_list)

class PermutationTest:
    def __init__(self, x, y, x_label, y_label, n_comparisons=1, nreps=10000,
                 seed=10):
        self.x, self.y = x.reshape((-1,1)), y.reshape((-1,1))
        self.x_label, self.y_label = x_label, y_label
        self.n_comparisons = n_comparisons
        self.stats, self.samples = permutation_test(
                x,y,difference_of_means, nreps=nreps, seed=seed)
        self.stats['p_val'] = min([self.stats['p_val']*n_comparisons,1])
        direction = self.stats.pop("direction",None)
        self.string = f'mean({x_label}){direction}mean({y_label})'

    def __str__(self):
        str_list = [f'Permutation test: {self.string}\n'] + \
               [f'\t{key}={value}\n' for key,value in self.stats.items()]
        return ''.join(str_list)

class KaplanMeierStatistics:
    def __init__(self, kaplan_meier_dict, name, key_1='Top', key_2='Bottom'):
        self.kaplan_meier_models = kaplan_meier_dict
        self.name = name
        self._merge_df(key_1, key_2)
        self.cph = CoxPHFitter().fit(self.survival_df, 'TIMES', 'PROGRESSES')
        self.pht = lifelines.statistics.proportional_hazard_test(
                self.cph, self.survival_df)
        print(self.cph.summary[['exp(coef)','exp(coef) lower 95%','exp(coef) upper 95%', 'p']])

    def _merge_df(self, key_1, key_2):
        top = self.kaplan_meier_models[key_1]
        bottom = self.kaplan_meier_models[key_2]
        top['Quartile'] = 1
        bottom['Quartile'] = 0
        self.survival_df = pd.concat([top, bottom], ignore_index=True)

    def __str__(self):
        return f'----------------\n{self.name} CPH model:\n' \
            '\tProportional hazard test: \n' + \
                    str(self.pht) + '\n' + \
                    '\tCPH coefficient' + str(self.cph.summary)

class KaplanMeierPairwise:
    def __init__(self, kaplan_meier_df, name, grouper_col='Cluster Idx', time=None):
        self.name = name
        self._merge_df(kaplan_meier_df.reset_index().copy(), grouper_col)
        new_col = grouper_col.replace(' ', '')
        self.survival_df.rename(columns={grouper_col: new_col}, inplace=True)
        self._compute_cph_values(new_col, time)

    def _compute_cph_values(self, grouper_col, time):
        categories = self.survival_df[grouper_col].cat.categories
        self.cph, self.pht = {}, {}
        self.n_comparisons = comb(len(categories),2)
        for cat in self.survival_df[grouper_col].cat.categories:
            surv_df = pd.DataFrame(data=self.survival_df.copy())
            surv_df[grouper_col] = surv_df[grouper_col].cat.reorder_categories(
                np.roll(categories, int(cat)))
            self.cph[cat] = CoxPHFitter()
            self.cph[cat].fit(surv_df, 'TIMES', 'PROGRESSES', formula=f'{grouper_col}')
            self.pht[cat] = lifelines.statistics.proportional_hazard_test(self.cph[cat], surv_df)
        
    def _merge_df(self, df, grouper):
        df_list = []
        for cluster, sub_df in df.groupby(grouper):
            df_list.append(sub_df[[grouper] + ['TIMES', 'PROGRESSES']])
        self.survival_df = pd.concat(df_list, axis=0, ignore_index=True)

    def __str__(self):
        output = '-'*100
        for i in self.cph.keys():
            self.cph[i].summary.loc['p'] = self.cph[i].summary['p']*self.n_comparisons
            output += '\nProportional hazard test:\n'
            output += str(self.pht[i]) + '\n'
            output += f'\nModel summary, bonferroni-corrected p-values {self.n_comparisons}\n'
            output += self.cph[i].summary.to_string(columns=['coef','se(coef)','z','p'])
            output += '\n\n-------------\n\n'
        output += '-'*100
        return output

def permutation_test(x,y, function, nreps=10000, seed=10):
    np.random.RandomState(seed)
    x = np.asarray(x).reshape(-1,1)
    n_x_nan = np.sum(np.isnan(x))
    x = x[~np.isnan(x)]
    mn_x = np.mean(x)
    n_x = x.shape[0]
    y = np.asarray(y).reshape(-1,1)
    n_y_nan = np.sum(np.isnan(y))
    y = y[~np.isnan(y)]
    n_y = y.shape[0]
    mn_y = np.mean(y)
    true_statistic = function(x,y)
    statistic_list = np.empty((nreps,1))
    all_data = np.concatenate([x,y],axis=0).squeeze()
    mn_all = np.mean(all_data)
    for rep in range(nreps):
        rand_x = np.random.choice(x, size=x.shape,
                                       replace=True)-mn_x + mn_all
        rand_y = np.random.choice(x, size=x.shape,
                                       replace=True)-mn_y + mn_all
        statistic_list[rep] = function(rand_x,rand_y)
    upper_tail = 2*np.sum(statistic_list >= true_statistic)/nreps
    lower_tail = 2*np.sum(statistic_list <= true_statistic)/nreps
    if upper_tail < 0.05:
        string = ">"
    elif lower_tail < 0.05:
        string = "<"
    else:
        string = "="
    return {
            "p_val": min([upper_tail, lower_tail]),
            "n_x": n_x,
            "n_y": n_y,
            "n_y_nan": n_y_nan,
            "n_x_nan": n_x_nan,
            "point_estimate": true_statistic,
            "direction": string
    }, statistic_list

def difference_of_medians(x,y):
    return np.median(x)-np.median(y)

def difference_of_means(x,y):
    return (np.mean(x)-np.mean(y))/np.sqrt(np.var(x, ddof=1)/x.shape[
        0]+np.var(y, ddof=1)/y.shape[0])

def pairwise_mannwhitneyu(df, col, group_col):
    factors = pd.unique(df[group_col])
    n_pairs = len(list(itertools.combinations(factors,2)))
    factor_pairs = itertools.combinations(factors,2)
    mwu_output_strings = []
    for pair in factor_pairs:
        pair_1, pair_2 = pair
        x = df.loc[df[group_col] == pair_1, col].copy().to_numpy()
        y = df.loc[df[group_col] == pair_2, col].copy().to_numpy()
        mwu = MannWhitneyU(x,y, pair_1, pair_2, n_comparisons=n_pairs)
        mwu_output_strings.append(str(mwu))
    return mwu_output_strings

def pairwise_perm(df, col, group_col):
    factors = pd.unique(df[group_col])
    n_pairs = len(list(itertools.combinations(factors,2)))
    factor_pairs = itertools.combinations(factors,2)
    pt_output_strings = ['']
    for pair in factor_pairs:
        pair_1, pair_2 = pair
        x = df.loc[df[group_col] == pair_1, col].copy().to_numpy()
        y = df.loc[df[group_col] == pair_2, col].copy().to_numpy()
        pt = PermutationTest(x,y, pair_1, pair_2, n_comparisons=n_pairs)
        print(pt)
        pt_output_strings.append(str(pt))
    return pt_output_strings

def chisq(tbl, nreps=1):
    chi2, p, dof, expected = stats.chi2_contingency(tbl)
    return {
            'chi2': chi2,
            'p': p*nreps,
            'dof': dof,
            'expected_lt5': any(expected.reshape((-1,1)) < 5)
    }

def _stringify_chisq(chisq_dict):
    return ''.join([f'\t{str(key)}={value}\n' for key,value in
                    chisq_dict.items()])

def compute_jaccard_similarity(nested_kmf_dict):
    region_combos = itertools.combinations(list(nested_kmf_dict.keys()),2)
    ic(region_combos)
    jaccard_idx = defaultdict(dict)
    ji = lambda x, y: len(np.intersect1d(x,y))/len(np.union1d(x,y))
    _str = []
    for combo in region_combos:
        region_x = nested_kmf_dict[combo[0]]
        region_y = nested_kmf_dict[combo[1]]
        _str.append(f'{combo[0]} and {combo[1]}')
        for quartile in region_x.keys():
            x = region_x[quartile]
            y = region_y[quartile]
            jaccard_idx[quartile][combo] = ji(x,y)
            _str.append(f'\t{quartile}: {jaccard_idx[quartile][combo]}')
    _str = '\n'.join(_str) + '\n'
    mn_bottom = np.mean(list(jaccard_idx['Bottom'].values()))
    mn_top = np.mean(list(jaccard_idx['Top'].values()))
    _str += f'\tmean, bottom quartile: {mn_bottom}\n'
    _str += f'\tmean, top quartile: {mn_top}\n'
    return jaccard_idx, _str

def kaplan_meier_estimator(df, label=''):
    if not all([x in df.columns for x in
                     ['TIMES', 'PROGRESSES']]):
        raise ValueError('Missing columns!')
    kmf = KaplanMeierFitter()
    times, hits = df.TIMES.to_numpy(), df.PROGRESSES.to_numpy()
    kmf.fit(times, hits, label=label)
    return kmf

