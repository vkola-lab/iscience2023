# icc's D ---------------------------------------------------------------
library("vcd")
library("irr")
library("tidyr")
library("readr")
library("dplyr")
library("reshape2")
# load table with individual ratings

tb.handed <- read_csv('/Users/mromano/research/mri-surv-dev/mri_surv/metadata/data_raw/MCIADSubtypeAssessment_handed.csv',
                      col_types = cols_only(
                        mesial_temp_l_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        mesial_temp_r_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        temporal_lobe_other_l_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        temporal_lobe_other_r_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        insula_l_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        insula_r_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        frontal_l_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        frontal_r_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        cingulate_l_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        cingulate_r_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        occipital_l_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        occipital_r_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        parietal_l_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        parietal_r_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        id=col_factor(),
                        rev_initials=col_factor()
                      ))
decoder <- read_csv('/Users/mromano/research/mri-surv-dev/mri_surv/metadata/data_processed/shuffled_mri_names.csv',
                    col_types=cols_only(
                      '...1'=col_factor(),
                      'Cluster Idx'=col_factor()
                    ))
tb.handed <- tb.handed %>%
  rowwise() %>%
  mutate(Cluster=(decoder[['Cluster Idx']][decoder[['...1']] == id]))

brain.regions = c('mesial_temp_l_sum',
                  'mesial_temp_r_sum',
                  'temporal_lobe_other_l_sum',
                  'temporal_lobe_other_r_sum',
                  'insula_l_sum',
                  'insula_r_sum',
                  'frontal_l_sum',
                  'frontal_r_sum',
                  'cingulate_l_sum',
                  'cingulate_r_sum',
                  'occipital_l_sum',
                  'occipital_r_sum',
                  'parietal_l_sum',
                  'parietal_r_sum')

# iterate through each column
# make tibble 

icc.list <- vector("list", length(brain.regions)+1)
for (i in 1:length(brain.regions)) {
  reg <- brain.regions[[i]]
  tb.current <- tb.handed[c(reg, 'rev_initials', 'id')]
  tb.current[[reg]] <- as.numeric(tb.current[[reg]])
  tb.current <- tb.current %>%
    spread(rev_initials, reg) %>%
    select(-id)
  icc.coef <- icc(tb.current, model='twoway', type='agreement', unit='single')
  nm.list <- c(unlist(icc.coef))
  nm.list$region <- reg
  icc.list[[i]] <- nm.list
}

tb.current <- tb.handed %>%
  select(-Cluster) %>%
  reshape2::melt(id=c('rev_initials', 'id')) %>%
  mutate(value=as.numeric(value)) %>%
  spread(rev_initials, value) %>%
  select(-c(variable,id))
icc.coef <- icc(tb.current, model='twoway', type='agreement', unit='single')
nm.list <- unlist(icc.coef)
nm.list$region <- "All"
icc.list[[length(brain.regions)+1]] <- nm.list


library("data.table")

# library(data.frame)
g <- data.table::rbindlist(icc.list)
g <- as_tibble(g)
write_csv(g, 'interrater_reliability_results.csv')