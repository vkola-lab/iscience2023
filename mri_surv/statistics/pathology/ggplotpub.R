library(ggplot2)
library(ggpubr)
library(rstatix)
library(tidyr)

theme_Publication <- function(base_size = 14, base_family = "helvetica") {
    library(grid)
    library(ggthemes)
    (theme_foundation(base_size = base_size, base_family = base_family)
    + theme(
            plot.title = element_text(
                face = "bold",
                size = rel(1.2), hjust = 0.5, family = "Montserrat"
            ),
            text = element_text(family = "Montserrat"),
            panel.background = element_rect(colour = NA),
            plot.background = element_rect(colour = NA),
            panel.border = element_rect(colour = NA),
            axis.title = element_text(face = "bold", size = rel(1)),
            axis.title.y = element_text(angle = 90, vjust = 2),
            axis.title.x = element_text(vjust = -0.2),
            axis.text = element_text(),
            axis.line = element_line(colour = "black"),
            axis.ticks = element_line(),
            panel.grid.major = element_line(colour = "#f0f0f0"),
            panel.grid.minor = element_blank(),
            legend.key = element_rect(colour = NA),
            legend.position = "bottom",
            legend.direction = "horizontal",
            legend.key.size = unit(0.2, "cm"),
            legend.margin = unit(0, "cm"),
            legend.title = element_text(face = "italic"),
            plot.margin = unit(c(10, 5, 5, 5), "mm"),
            strip.background = element_rect(colour = "#f0f0f0", fill = "#f0f0f0"),
            strip.text = element_text(face = "bold")
        ))
}

do_stats <- function(df) {
    df %>% pairwise_fisher_test(p.adjust.method = "hochberg")
}

scale_fill_Publication <- function(...) {
    library(scales)
    discrete_scale("fill", "Publication", manual_pal(values = c("#386cb0", "#fdb462", "#7fc97f", "#ef3b2c", "#662506", "#a6cee3", "#fb9a99", "#984ea3", "#ffff33")), ...)
}

scale_colour_Publication <- function(...) {
    library(scales)
    discrete_scale("colour", "Publication", manual_pal(values = c("#386cb0", "#fdb462", "#7fc97f", "#ef3b2c", "#662506", "#a6cee3", "#fb9a99", "#984ea3", "#ffff33")), ...)
}

braak <- read.csv("./braak.csv")
braak_long <- braak %>% pivot_longer(Stage.0:Stage.6, names_to = "Names", values_to = "Values")
braak_long$Names <- sub("\\.", " ", braak_long$Names)
rownames(braak) <- braak$Clinical.progression.
braak$Clinical.progression. <- NULL
names(braak) <- sub("\\.", " ", names(braak))

cerad <- read.csv("./cerad.csv")
cerad_long <- cerad %>% pivot_longer(C0:C3, names_to = "Names", values_to = "Values")
rownames(cerad) <- cerad$Clinical.progression.
cerad$Clinical.progression. <- NULL

adnc <- read.csv("./nia_adnc.csv")
adnc_long <- adnc %>% pivot_longer(Not.AD:High, names_to = "Names", values_to = "Values")
adnc_long$Names[which(adnc_long$Names == "Not.AD")] <- "Not AD"
rownames(adnc) <- adnc$Clinical.progression.
adnc$Clinical.progression. <- NULL
names(adnc)[1] <- "Not AD"


names(braak_long)[1] <- "Progression"
names(cerad_long)[1] <- "Progression"
names(adnc_long)[1] <- "Progression"


# doing stats here
braak.stats <- do_stats(braak) %>% mutate(y.position = 1.03)
adnc.stats <- do_stats(adnc) %>% mutate(y.position = 1.03)
cerad.stats <- do_stats(cerad) %>% mutate(y.position = 1.03)

braak.stats$y.position[4] <- 1.1


# fig <- ggplot(braak_long, aes(y=Values, x=Names)) + geom_bar(aes(fill=Progression), position='fill', stat='identity') + xlab("BRAAK Stage") + ylab("Percent") + scale_colour_Publication() + theme_Publication() + scale_y_continuous(labels=scales::percent, breaks=c(.25, .50, .75, 1.00)) + scale_fill_manual(values=c("#36887A", "#822B3A")) + stat_pvalue_manual(braak.stats, hide.ns=TRUE, label="p.adj.signif", tip.length=0.001)
# ggsave('braak.eps', fig, width=15, height=10, units="cm")

# fig <- ggplot(adnc_long, aes(y=Values, x=Names)) + geom_bar(aes(fill=Progression), position='fill', stat='identity') + xlab("ADNC") + ylab("Percent") + scale_colour_Publication() + theme_Publication() + scale_y_continuous(labels= scales::percent) + scale_fill_manual(values=c("#36887A", "#822B3A")) + scale_x_discrete(limits=c("Not AD", "Low", "Intermediate", "High")) + stat_pvalue_manual(adnc.stats, hide.ns=TRUE, label="p.adj.signif", tip.length=0.001)
# ggsave('adnc.eps', fig, width=15, height=10, units = "cm")

# fig <- ggplot(cerad_long, aes(y=Values, x=Names)) + geom_bar(aes(fill=Progression), position='fill', stat='identity') + xlab("CERAD score") + ylab("Percent") + scale_colour_Publication() + theme_Publication() + scale_y_continuous(labels=scales::percent) + scale_fill_manual(values=c("#36887A", "#822B3A")) + stat_pvalue_manual(cerad.stats, hide.ns=TRUE, label="p.adj.signif", tip.length=0.001)
# ggsave('cerad.eps', fig, width=15, height=10, units = "cm")
