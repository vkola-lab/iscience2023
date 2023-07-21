# init ----------------------------
library(dplyr)
library(tibble)
library(readr)
library(tidyr)
library(broom)
library(ggplot2)
library(gridExtra)
library(sjPlot)
library(purrr)

if (Sys.info()['sysname'] == "Linux") {
    setwd('/home/mfromano/Research/mri-surv-dev/mri_surv')
} else {
    setwd('/Users/mromano/research/mri-surv-dev/mri_surv/')
}

bar.tibble <- read_csv('./metadata/data_processed/masked_shap_brains_rid_lobe_mn_abs.csv',
               col_select = c('Region','Shap','Lobe','RID','Cluster'),
               col_types = cols(
                 Region=col_factor(),
                 Shap=col_number(),
                 Lobe=col_factor(),
                 Cluster=col_factor(),
                 RID=col_factor()
               ))

bar.tibble$Lobe = recode(
  bar.tibble$Lobe,
 'Basal-Ganglia'='BG',
 'Subcortical'='SC',
 'Insula'='Ins',
 'TL'='TL-O', 
 'Limbic-Cing'='Cing'
 )

bar.tibble$Cortex <- bar.tibble$Lobe

bar.tibble$Cortex <- as.character(bar.tibble$Cortex)

lev <- unique(bar.tibble$Cortex)

bar.tibble$Cortex <- factor(
    bar.tibble$Cortex)

bar.tibble$Lobe <- factor(
    bar.tibble$Lobe)

write_csv(bar.tibble, './metadata/data_processed/shap_cnn_abs_barplot.csv')

# Now plot ---------------------------------------------------

library(rsample)

library(RColorBrewer)
colors <- brewer.pal(10,'Paired')

cortex.names <- c("Cing", "Ins", "BG", "PL", "TL-M", "FL", "SC", "TL-O", "OL")

for (subtype in unique(bar.tibble$Cluster)) {
  tb <- bar.tibble %>%
    filter(Cluster == subtype) %>%
    drop_na() %>%
    select(-Cluster, -Region, -Cortex, -RID) %>%
    ungroup()

  st <- tb %>%
    group_by(Lobe) %>%
    summarize(Dat=mean_cl_boot(Shap, conf.int=.95, B=10000))
  print(st)
}

bar.tibble$Cluster <- factor(bar.tibble$Cluster, levels = c('L','IL','IH','H'))

  ggobj <- ggplot(bar.tibble, aes(x=Lobe, y=Shap, fill=Lobe)) +
    facet_grid(cols=vars(Cluster)) +
    stat_summary(fun=mean, geom="bar", na.rm=FALSE, show.legend=TRUE) +
    stat_summary(fun.data=mean_cl_boot, fun.args=list(conf.int=.95, B=10000), size=1, geom = "linerange")+
    theme(
      axis.text.y = element_text(size=12),
      axis.title.y.left = element_text(size=12),
      strip.text.x = element_text(size=12, face= 'bold'),
      legend.text = element_text(size=10),
      legend.title = element_text(size=12),
      axis.text.x = element_blank(),
      panel.background = element_blank(),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.ticks.x=element_blank(),
      aspect.ratio = 1.5
    ) + # coord_cartesian(ylim=c(-4e-06, 5e-06))+
    xlab('')+
   scale_fill_manual(values = colors, breaks=cortex.names)
  ggsave(plot = ggobj, paste0('./figures/bar_cnn_abs_all.svg'),
         device = 'svg', dpi='retina', width=10.0, height=6.68, units='in')