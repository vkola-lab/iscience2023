# load things -----
library(dplyr)
library(tibble)
library(readr)
library(tidyr)
library(reshape2)
library(stringr)

if (Sys.info()["sysname"] == "Linux") {
    setwd("~/Research/mri-surv-dev/mri_surv/")
} else {
    setwd("C://Users/micha/Sync/Research/mri-pet/mri_surv/")
}
tb <- read_csv("./metadata/data_raw/MCIADSubtypeAssessment_weighted_nosub.csv",
                 col_types = cols(
                 .default = col_factor(),
                 mesial_temp_avg = col_number(),
                 temporal_lobe_other_avg = col_number(),
                 insula_avg = col_number(),
                 frontal_avg = col_number(),
                 cingulate_avg = col_number(),
                 occipital_avg = col_number(),
                 parietal_avg = col_number(),
                 id = col_factor(),
                 id2 = col_factor(),
                 rev_initials = col_factor()
               ))

decoder <- read_csv("./metadata/data_processed/shuffled_mri_names.csv")

# reorganize data w/in table -------------------------------------

tb <- tb %>%
  rowwise() %>%
  mutate(Subtype = (decoder[["Cluster Idx"]][decoder[["...1"]] == id]))

tb$SubtypeSymbol <-
    as.factor(recode(
        tb$Subtype, "0" = "H", "1" = "IH", "2" = "IL", "3" = "L"))

tb$Subtype <- 3 - tb$Subtype

tb <- tb %>%
  mutate(Reviewer = rev_initials) %>%
  dplyr::select(-c(rev_initials))

tb$Reviewer <-
    recode(
        tb$Reviewer,
        "ABP" = "1",
        "AZM" = "2",
        "JES" = "3",
        "MJS" = "4",
        "PHL" = "5")

tb <- tb %>%
  dplyr::select(-c(id2, ...1))

tb <- tb %>%
  pivot_longer(
    cols = -c(Reviewer, SubtypeSymbol, Subtype, id),
    names_to = "Region",
    values_to = "Grade"
    )

tb$Region <- as.factor(
  recode(
    tb$Region,
    "cingulate_avg" = "Cingulate",
    "frontal_avg" = "Frontal",
    "insula_avg" = "Insula",
    "mesial_temp_avg" = "Mesial Temporal",
    "temporal_lobe_other_avg" = "Temporal (other)",
    "occipital_avg" = "Occipital",
    "parietal_avg" = "Parietal"
    ))

# Take the average grade for each entire lobe within each cluster, plot it

library(ggplot2)
library(svglite)
#trying to loop over each lobe, manually change the reviewer


ggplot(tb) + aes_string(x = "Subtype", y = "Grade", fill = "Reviewer") +
  geom_jitter(height = 0, width = 0.2,
    aes(colour = Reviewer)) +
    coord_cartesian(ylim = c(0, 3.0)) +
    geom_smooth(formula = y ~ x, method = "lm", inherit.aes = TRUE) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black")
  ) +  # help from https://felixfan.github.io/ggplot2-remove-grid-background-margin/
  theme(axis.text.x = element_text(
    colour = "black", size = 15),
    axis.title.x = element_text(
      colour = "black", size = 20
    )) +
  theme(axis.text.y = element_text(
    colour = "black", size = 15),
    axis.title.y=element_text(
      colour="black", size=20
    )) +
  theme(strip.text = element_text(
    colour = "black", size = 15)) +
  theme(legend.title = element_text(colour="black",size=15), 
        legend.position=c(0.7, 0.1),
        legend.direction="horizontal",
        legend.text = element_text(colour="black", size=12) )+
  facet_wrap(~Region)

ggsave(file=paste0("figures/reviewer_grades_new.svg"), width=8, height=10)
