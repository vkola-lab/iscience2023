# init ----------------------------
library(languageserver)
library(dplyr)
library(tibble)
library(readr)
library(tidyr)
library(lme4)
library(lmerTest)
library(broom)
library(emmeans)
library(ggplot2)
library(gridExtra)
library(sjPlot)

setwd('C://Users/micha/Sync/Research/mri-pet/mri_surv/')

##########################################
########Now by lobe

tb2 <- read_csv(
  './metadata/data_processed/cortical_regions_by_subtype.csv',
                col_types = cols('Subtype'=col_factor(),'RID'=col_factor(),'Cortical Region'=col_factor(), 'Visit'=col_factor(),
                                 'AGE'=col_number(), 'SEX'=col_factor()))

tb2 <- rename(tb2, 'Cortex'='Cortical Region', 'GMV'='ZS Gray Matter Volume')
tb2$Subtype <- recode(tb2$Subtype, 'H'='H','I-H'='IH','I-L'='IL','L'='L')

mdl2 <- lmer('GMV ~ Cortex*Subtype*Visit + AGE + SEX + (Cortex|RID)', tb2, verbose=TRUE)
anova(mdl2)

mdl2 <- lmer('GMV ~ Cortex*Subtype*Visit-Cortex:Subtype:Visit +AGE+SEX+ (Cortex|RID)', tb2)
anova(mdl2)

sink('./results/cortex_lme_cortex_by_subtype_contrasts.txt')
print(anova(mdl2))
print(summary(mdl2))
sink()

results <- lsmeans(mdl2, pairwise~Cortex|Subtype, adjust="sidak", at=c(Visit='MCI'))
summ <- summary(results)
write_tsv(data.frame(summ$lsmeans), './results/cortex_lme_cortex_by_subtype_means.txt')
write_tsv(data.frame(summ$contrasts), './results/cortex_lme_cortex_by_subtype_contrasts.txt')

results <- lsmeans(mdl2, pairwise~Subtype|Cortex, adjust="sidak", at=c(Visit='MCI'))
summ <- summary(results)
write_tsv(data.frame(summ$lsmeans), './results/cortex_lme_subtype_by_cortex_means.txt')
write_tsv(data.frame(summ$contrasts), './results/cortex_lme_subtype_by_cortex_contrasts.txt')

results <- lsmeans(mdl2, pairwise~Visit, adjust="sidak", at=c(Subtype='H'))
summ <- summary(results)
write_tsv(data.frame(summ$lsmeans), './results/cortex_lme_visit_means.txt')
write_tsv(data.frame(summ$contrasts), './results/cortex_lme_visit_contrasts.txt')

# non-parametrics -------------------
library(qqplotr)
tb2$Subtype <- ordered(tb2$Subtype, levels=c('H','IH','IL','L'))

res <- residuals(mdl2,'pearson')

plt1 <- ggplot(data.frame(Residuals=res, Subtype=tb2$Subtype),
               aes(x=Subtype, y=Residuals)) +
  geom_boxplot(notch=TRUE)

plt2 <- ggplot(data.frame(Residuals=res, Cortex=tb2$Cortex),
               aes(x=Cortex, y=Residuals)) +
  geom_boxplot(notch=TRUE)

plt5 <- ggplot(data.frame(Residuals=res, Visit=tb2$Visit),
               aes(x=Visit, y=Residuals)) +
  geom_boxplot(notch=TRUE)

names(res) <- NULL

plt3 <- ggplot(data.frame(Residuals=(res)), aes(sample=Residuals)) +
  stat_qq_band() +
  stat_qq_line() +
  stat_qq_point()

plt4 <- ggplot(data.frame(Fitted=fitted(mdl2),Residuals=res),
               aes(x=Fitted,y=Residuals)) +
  geom_point() +
  theme_bw()

p <- grid.arrange(plt1, plt2, plt5, plt3, plt4, nrow=2)

ggsave('./figures/lme_diagnostics3.svg',plot=p, device='svg', dpi='retina')




