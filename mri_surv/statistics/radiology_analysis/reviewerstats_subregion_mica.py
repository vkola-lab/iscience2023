import pandas as pd
import matplotlib.pyplot as plt
import sys
import re
import math

"""
need to check
if there is atrophy there should be no nan values

"""

# start from mri_pet working directory
csv = pd.read_csv("./metadata/data_raw/MCIADSubtypeAssessment.csv")

# Need to create a tree of a loop

lobes = [
    "mesial_temp",
    "temporal_lobe_other",
    "insula",
    "frontal",
    "cingulate",
    "occipital",
    "parietal",
]

lobe_dict = {
    "mesial_temp": ["mesial_temp"],
    "temporal_lobe_other": ["temporal"],
    "insula": ["insula"],
    "frontal": ["frontal"],
    "cingulate": ["cingulate"],
    "occipital": ["occipital"],
    "parietal": ["parietal"],
}

# --> coming up with a sum of the _l and _r subregions for each
# loop over every row
for index, row in csv.iterrows():
    for lobe in lobes:
        l_sum = []  # once filled, take a sum of this list and add to dataframe
        r_sum = []  # once filled, take a sum of this list and add to dataframe
        # lobe_sum = [] #once filled, this is the sum of l_sum and r_sum

        # walk down each row and create a weighted sum across the entire lobe
        # and also the l and r
        # add columns for each lobe... l_sum, r_sum, lobe_sum
        # --> l_sum = the sum over all the l subregions
        # --> r_sum = the sum over all the r subregions
        # --> lobe_sum = the sum over l and right subregions (all)
        if row[lobe] == 0:
            # append a 0 for l_sum, r_sum, and lobe_sum
            l_sum.append(0)
            r_sum.append(0)
            # lobe_sum = 0
        else:
            for subregion in lobe_dict[lobe]:  # list of the subregions
                # get a l_sum and r_sum for each subregion,
                # append to the *_sum_subregions for this lobe
                l_sum_subregion = 0
                r_sum_subregion = 0

                l_sum_subregion += row["l_" + subregion]

                l_sum.append(l_sum_subregion)

                r_sum_subregion += row["r_" + subregion]

                r_sum.append(r_sum_subregion)

        assert len(l_sum) == 1 and len(r_sum) == 1

        l_sum = sum(l_sum)  # exclude the first one
        r_sum = sum(r_sum)  # exclude the first one
        lobe_avg = (l_sum + r_sum) / (len(lobe_dict[lobe]) * 2)
        csv.loc[index, lobe + "_l_sum"] = l_sum
        csv.loc[index, lobe + "_r_sum"] = r_sum
        csv.loc[index, lobe + "_avg"] = lobe_avg

salient_columns = list(filter(lambda x: re.match(r".*_avg", x), csv.columns))
salient_columns += ["id", "id2", "rev_initials"]

handed_columns = list(filter(lambda x: re.match(r".*[lr]{1}_sum", x), csv.columns))
handed_columns += ["id", "rev_initials"]

csv2 = csv[salient_columns]
# print(csv['mesial_temp_avg'])
csv2.to_csv("./metadata/data_raw/MCIADSubtypeAssessment_weighted_nosub.csv")

csv_handed = csv[handed_columns]
csv_handed.to_csv("./metadata/data_raw/MCIADSubtypeAssessment_handed.csv", index=False)
