# init ----------------------------
library(dplyr)
library(tibble)
library(readr)
library(tidyr)
library(broom)
library(ggplot2)
library(gridExtra)
library(sjPlot)
library(FSA)
setwd('C://Users/micha/Sync/Research/mri-pet/mri_surv/')

##########################################
########Now by lobe

tb2 <- read_csv('./metadata/data_processed/cortical_regions_by_subtype.csv',
                col_types = cols('Subtype'=col_factor(),'RID'=col_factor(),'Cortical Region'=col_factor(), 'Visit'=col_factor(),
                                 'AGE'=col_number(), 'SEX'=col_factor()))

tb2 <- rename(tb2, 'Cortex'='Cortical Region', 'GMV'='ZS Gray Matter Volume')
tb2$Subtype <- recode(tb2$Subtype, 'H'='H','I-H'='IH','I-L'='IL','L'='L')

mci.tb <- tb2 %>%
  filter(Visit == 'MCI') %>%
  group_by(RID, Subtype) %>%
  summarize(GMV=mean(GMV)) %>%
  ungroup()

summ <- Summarize(GMV~Subtype, mci.tb)
mci.st <- kruskal.test(GMV~Subtype, mci.tb)

PT <- dunnTest(GMV~Subtype, mci.tb, method = 'bonferroni')

sink('./results/gmv_by_subtype_mci.txt')
print(mci.st)
print(PT)
sink()

ad.tb <- tb2 %>%
  filter(Visit == 'AD') %>%
group_by(RID, Subtype) %>%
  summarize(GMV=mean(GMV)) %>%
  ungroup()

summ <- Summarize(GMV~Subtype, ad.tb)
ad.st <- kruskal.test(GMV~Subtype, ad.tb)

PT.ad <- dunnTest(GMV~Subtype, ad.tb, method = 'bonferroni')


sink('./results/gmv_by_subtype_ad.txt')
print(ad.st)
print(PT.ad)
sink()

mci.tb <- tb2 %>%
  filter(Visit == 'MCI')
comparisons <- c('L','IL','IH','H')
mci.tb$Subtype <- factor(mci.tb$Subtype, comparisons)
p <- c()
stat <- c()
est <- c()
comparison <- c()
lobe <- c()

for (c in 1:(length(comparisons)-1)) {
  sub.tbl <- mci.tb %>%
    filter(Visit == 'MCI', Subtype %in% c(comparisons[c], comparisons[c+1]))
  sub.tbl$Subtype <- droplevels(sub.tbl$Subtype)
  for (region in unique(sub.tbl$Cortex)) {
    sub.sub.tbl <- sub.tbl %>%
      filter(Cortex == region)
    x = sub.sub.tbl$GMV[sub.sub.tbl$Subtype==comparisons[c]]
    y = sub.sub.tbl$GMV[sub.sub.tbl$Subtype==comparisons[c+1]]
    st <- wilcox.test(x,y, conf.int=TRUE)
    p <- c(p, st$p.value)
    stat <- c(stat, st$statistic)
    est <- c(est, st$estimate)
    comparison <- c(comparison, paste0(comparisons[c], '-',comparisons[c+1]))
    lobe <- c(lobe, region)
    }
}

names(stat) <- NULL
names(est) <- NULL
wil.stats <- tibble(lobe, comparison, est, stat, p)
 
wil.stats$p.correct <- p.adjust(wil.stats$p, method="fdr")

write_tsv(wil.stats, file='./results/cortical_gmv_comparisons.tsv')

comparisons <- c('MCI', 'AD')
tb2$Visit <- factor(tb2$Visit, comparisons)
p <- c()
stat <- c()
est <- c()
comparison <- c()
lobe <- c()

for (region in unique(tb2$Cortex)) {
    sub.tbl <- tb2 %>%
      filter(Cortex == region)
    ad.rids <- unique(sub.tbl$RID[sub.tbl$Visit == 'AD'])
    sub.tbl <- sub.tbl %>%
      filter(RID %in% ad.rids) %>%
      droplevels() %>%
      arrange(RID)
    
    x <- sub.tbl %>% filter(Visit == 'MCI') %>% arrange(RID)
    y <-  sub.tbl %>% filter(Visit == 'AD') %>% arrange(RID)
    st <- wilcox.test(x$GMV,y$GMV, paired = TRUE, conf.int=TRUE)
    p <- c(p, st$p.value)
    stat <- c(stat, st$statistic)
    est <- c(est, st$estimate)
    lobe <- c(lobe, region)
}

names(stat) <- NULL
names(est) <- NULL
wil.stats <- tibble(lobe, est, stat, p)

wil.stats$p.correct <- p.adjust(wil.stats$p, method="fdr")

write_tsv(wil.stats, file='./results/cortical_visit_comparisons.tsv')
