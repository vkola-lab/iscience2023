# init ----------------------------
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
#  0.34137757667169427 +/- 0.10511520840922657 overlap
setwd('C://Users/micha/Sync/Research/mri-pet/mri_surv/')


# try using non-parametrics 
tb <- read_csv('./results/shap_dataframe.csv',
               col_select = c('Cluster Idx', 'Region','RID','Shap Value','Gray Matter Vol', 'Cortex'),
               col_types = cols(
                 Region=col_factor(),
                 RID=col_factor(),
                 Dataset=col_factor(),
                 'Shap Value'=col_number(),
                 'Gray Matter Vol'=col_number(),
                 Cortex=col_factor(),
                 'Cluster Idx'=col_factor()
               ))

tb$Cluster <- tb$`Cluster Idx`
tb$GMV <- tb$`Gray Matter Vol`
tb$Shap <- tb$`Shap Value`

waterfall.tibble <- tb %>%
  group_by(Cluster, Region, .add = TRUE) %>%
  summarize(Shap=mean(Shap), GMV=mean(GMV), Cortex=unique(Cortex)) %>%
  ungroup()

library(waterfalls)

waterfall.tibble$Cluster <- recode(
  waterfall.tibble$Cluster, '0'='H','1'='IH','2'='IL','3'='L'
  )

waterfall.tibble$Cortex <- as.character(waterfall.tibble$Cortex)

waterfall.tibble$Cortex <- recode(na_if(waterfall.tibble$Cortex,''), .missing ='Other')

waterfall.tibble$Cortex <- as.factor(waterfall.tibble$Cortex)

# Now plot ---------------------------------------------------

library(RColorBrewer)
colors <- brewer.pal(10,'Paired')

cortex.names <- unique(waterfall.tibble$Cortex)

for (subtype in unique(waterfall.tibble$Cluster)) {
  tb <- waterfall.tibble %>%
    filter(Cluster == subtype) %>%
    select(-Cluster) %>%
    group_by(Cortex) %>%
    mutate(ShapSum=abs(sum(Shap))) %>%
    arrange(desc(ShapSum), desc(abs(Shap))) %>%
    ungroup()
  
  tb$Rank <- 1:dim(tb)[1]
  tb$CumSumOffset <- c(0,cumsum(tb$Shap)[1:(length(tb$Shap)-1)])
  tb$CumSum <- cumsum(tb$Shap)
  tb <- tb %>%
    group_by(Cortex) %>%
    mutate(Min=min(CumSumOffset), Height=max(CumSumOffset)-min(CumSumOffset)) %>%
    ungroup()

  ggobj <- waterfall(
    tb, calc_total = TRUE,
    rect_text_labels = rep('', dim(tb)[1]),
    total_rect_text = '', ggplot_object_name = 'cluster')+
    theme(
      axis.title.x = element_text(size = 20),
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      axis.text.y = element_text(size=20),
      axis.title.y.left = element_text(size=20),
      plot.title = element_text(size=32, face = 'bold',hjust = .5),
      legend.text = element_text(size=20),
      legend.title = element_text(size=24)
          ) +
    xlab('Region')+
    ylab('Sum(SHAP)')+
    ggtitle(
      label=paste0(subtype))
  ggobj$layers <- c(
    geom_tile(
      data=tb, mapping=aes(x=Rank, y=Min+Height/2, fill=Cortex, width=1, height=Height), alpha=.6),
    ggobj$layers
  )
    scale_fill_manual(values = colors, breaks=cortex.names)
  ggsave(plot = ggobj, paste0('./figures/waterfall_plot_',subtype,'.svg'),device = 'svg')
  
  gg.bar.graph <- ggplot(tb, aes(x=Rank, y=GMV, fill=Cortex)) +
    geom_bar(width=1, stat='identity') +
    scale_color_manual(values=colors, breaks=cortex.names)+
    xlab('Region') +
    ylab('Gray Matter Volume')+
  theme(
    axis.title.x = element_text(size = 20),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.text.y = element_text(size=20),
    axis.title.y.left = element_text(size=20),
    plot.title = element_text(size=32, face = 'bold',hjust = .5),
    legend.text = element_text(size=20),
    legend.title = element_text(size=24)
  )
  ggsave(plot = gg.bar.graph, paste0('./figures/bargraph_',subtype,'.svg'),device = 'svg')
}

# now compute statistics ------

cor_tb <- waterfall.tibble %>%
  group_by(Cortex) %>%
  summarise(RegionCor=cor(Shap, GMV, method='spearman'), n=n())

write_tsv(cor_tb, './results/shap_gmv_cor.tsv')


# compute diffs in shap values for each cortex  ------------------------------
tb <- read_csv('./results/shap_dataframe.csv',
               col_select = c('Cluster Idx', 'Region','RID','Shap Value','Gray Matter Vol', 'Cortex'),
               col_types = cols(
                 Region=col_factor(),
                 RID=col_factor(),
                 Dataset=col_factor(),
                 'Shap Value'=col_number(),
                 'Gray Matter Vol'=col_number(),
                 Cortex=col_factor(),
                 'Cluster Idx'=col_factor()
               ))

tb$Cluster <- tb$`Cluster Idx`
tb$GMV <- tb$`Gray Matter Vol`
tb$Shap <- tb$`Shap Value`

# --------------

shap.tibble <- tb %>%
  group_by(Cluster, RID, .add = TRUE) %>%
  summarize(Shap=mean(Shap)) %>%
  ungroup()

shap.tibble$Cluster <- recode(
  shap.tibble$Cluster, '0'='H','1'='IH','2'='IL','3'='L'
)


mdl <- kruskal.test(Shap~Cluster, shap.tibble)

PT <- dunnTest(Shap~Cluster, shap.tibble, method = 'bonferroni')


sink('./results/shap_by_subtype_ad.txt')
print(mdl)
print(PT)
sink()


# Next, compare regions between subtypes -------------------------------------

shap.tibble <- tb %>%
  group_by(Cortex, RID, Cluster) %>%
  summarize(Shap=mean(`Shap Value`)) %>%
  ungroup()

shap.tibble$Cluster <- recode(
  shap.tibble$Cluster, '0'='H','1'='IH','2'='IL','3'='L'
)

comparisons <- c('L','IL','IH','H')
shap.tibble$Cluster <- factor(shap.tibble$Cluster, comparisons)
shap.tibble$Subtype <- shap.tibble$Cluster
p <- c()
stat <- c()
est <- c()
comparison <- c()
lobe <- c()

for (c in 1:(length(comparisons)-1)) {
  sub.tbl <- shap.tibble %>%
    filter(Subtype %in% c(comparisons[c], comparisons[c+1]))
  sub.tbl$Subtype <- droplevels(sub.tbl$Subtype)
  for (region in unique(sub.tbl$Cortex)) {
    sub.sub.tbl <- sub.tbl %>%
      filter(Cortex == region)
    x = sub.sub.tbl$Shap[sub.sub.tbl$Subtype==comparisons[c]]
    y = sub.sub.tbl$Shap[sub.sub.tbl$Subtype==comparisons[c+1]]
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

write_tsv(wil.stats, file='./results/cortical_shap_comparisons.tsv')
