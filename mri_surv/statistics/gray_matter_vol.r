library(dplyr)
library(tibble)
library(readr)
library(tidyr)
library(broom)
library(ggplot2)
library(gridExtra)
library(sjPlot)
library(FSA)
setwd('/Users/mromano/research/mri-surv-dev/mri_surv/')

##########################################
########Now by lobe

tb2 <- read_csv('./metadata/data_processed/shap_with_parcellations_long.csv',
                col_types = cols(`Cluster Idx`=col_factor(),'RID'=col_factor(),'Dataset'=col_factor(),
                                'Region'=col_factor(),
                                'GMV'=col_factor(),
                                ))


tb2$Subtype <- recode(tb2$`Cluster Idx`, 'H'='0','I-H'='1','I-L'='2','L'='3')

mci.tb <- tb2 %>%
  filter(Dataset == 'ADNI') %>%
  group_by(RID, Subtype) %>%
  summarize(GMV=mean(GMV)) %>%
  ungroup()

summ <- Summarize(GMV~Subtype, mci.tb)
mci.st <- kruskal.test(GMV~Subtype, mci.tb)

PT <- dunnTest(GMV~Subtype, mci.tb, method = 'bh')

sink('./results/gmv_by_subtype_mci_new_bh.txt')
print(mci.st)
print(PT)
sink()
