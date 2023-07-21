library(dplyr)
library(stringr)

DATA_DIR <- "/data2/MRI_PET_DATA/predicts/" # nolint

################################################################
####### Start MLP
################################################################

mlp_dat <- read.csv(paste0(DATA_DIR, "mlp_predictions.csv"))
mlp_dat$Dataset <- as.factor(mlp_dat$Dataset)
mlp_dat$RID <- as.character(mlp_dat$RID)

unique_rids <- unique(mlp_dat$RID)
nacc_order <- unique_rids[str_detect(unique_rids, "NACC")]
adni_order <- unique_rids[!str_detect(unique_rids, "NACC")]

mlp_dat$Experiment <- as.factor(mlp_dat$Experiment)

m0 <- mlp_dat[mlp_dat$Bins == 0, "Predictions"]
m24 <- mlp_dat[mlp_dat$Bins == 24, "Predictions"]
m48 <- mlp_dat[mlp_dat$Bins == 48, "Predictions"]
m108 <- mlp_dat[mlp_dat$Bins == 108, "Predictions"]

stopifnot(length(m24) == length(m48), length(m48) == length(m108))

exp_idx <- c()
experiment <- summary(mlp_dat[mlp_dat$Dataset == "ADNI", "Experiment"]) / 4

for (i in 0:4) {
    exp_idx <- c(exp_idx, rep(i, experiment[[i + 1]]))
}

experiment <- summary(mlp_dat[mlp_dat$Dataset == "NACC", "Experiment"]) / 4
for (i in 0:4) {
    exp_idx <- c(exp_idx, rep(i, experiment[[i + 1]]))
}

stopifnot(length(m24) == length(exp_idx))



mlp_dat <- data.frame(
    "RID" = mlp_dat[seq(1, nrow(mlp_dat), 4), "RID"],
    "Experiment" = exp_idx,
    "Dataset" = mlp_dat[seq(1, nrow(mlp_dat), 4), "Dataset"],
    "pred.prob.0" = m0,
    "pred.prob.24" = m24,
    "pred.prob.48" = m48,
    "pred.prob.108" = m108
)

# write experiemnt-wise
for (idx in 0:4) {
    write.csv(mlp_dat[mlp_dat$Experiment == idx & mlp_dat$Dataset == "ADNI", ],
        file = paste0(DATA_DIR, "mlp_exp", idx, "_ADNI_test.csv"),
        quote = FALSE,
        row.names = FALSE
    )

    write.csv(mlp_dat[mlp_dat$Experiment == idx & mlp_dat$Dataset == "NACC", ],
        file = paste0(DATA_DIR, "mlp_exp", idx, "_NACC.csv"),
        quote = FALSE,
        row.names = FALSE
    )
}

################################################################
####### Start MLP #2: csf and gmv values
################################################################

mlp_dat_gmv_csf <- read.csv(paste0(DATA_DIR, "mlp_parcellation_gmv_csf_sur_loss.csv"))
mlp_dat_gmv_csf$Dataset <- as.factor(mlp_dat_gmv_csf$Dataset)
mlp_dat_gmv_csf$RID <- as.character(mlp_dat_gmv_csf$RID)

unique_rids <- unique(mlp_dat_gmv_csf$RID)
nacc_order <- unique_rids[str_detect(unique_rids, "NACC")]
adni_order <- unique_rids[!str_detect(unique_rids, "NACC")]

mlp_dat_gmv_csf$Experiment <- as.factor(mlp_dat_gmv_csf$Experiment)

m0 <- mlp_dat_gmv_csf[mlp_dat_gmv_csf$Bins == 0, "Predictions"]
m24 <- mlp_dat_gmv_csf[mlp_dat_gmv_csf$Bins == 24, "Predictions"]
m48 <- mlp_dat_gmv_csf[mlp_dat_gmv_csf$Bins == 48, "Predictions"]
m108 <- mlp_dat_gmv_csf[mlp_dat_gmv_csf$Bins == 108, "Predictions"]

stopifnot(length(m24) == length(m48), length(m48) == length(m108))

exp_idx <- c()
experiment <- summary(mlp_dat_gmv_csf[mlp_dat_gmv_csf$Dataset == "ADNI", "Experiment"]) / 4

for (i in 0:4) {
    exp_idx <- c(exp_idx, rep(i, experiment[[i + 1]]))
}

experiment <- summary(mlp_dat_gmv_csf[mlp_dat_gmv_csf$Dataset == "NACC", "Experiment"]) / 4
for (i in 0:4) {
    exp_idx <- c(exp_idx, rep(i, experiment[[i + 1]]))
}

stopifnot(length(m24) == length(exp_idx))



mlp_dat_gmv_csf <- data.frame(
    "RID" = mlp_dat_gmv_csf[seq(1, nrow(mlp_dat_gmv_csf), 4), "RID"],
    "Experiment" = exp_idx,
    "Dataset" = mlp_dat_gmv_csf[seq(1, nrow(mlp_dat_gmv_csf), 4), "Dataset"],
    "pred.prob.0" = m0,
    "pred.prob.24" = m24,
    "pred.prob.48" = m48,
    "pred.prob.108" = m108
)

# write experiemnt-wise
for (idx in 0:4) {
    write.csv(mlp_dat_gmv_csf[mlp_dat_gmv_csf$Experiment == idx & mlp_dat_gmv_csf$Dataset == "ADNI", ],
        file = paste0(DATA_DIR, "mlp_exp_gmv_csf_exp", idx, "_ADNI_test.csv"),
        quote = FALSE,
        row.names = FALSE
    )

    write.csv(mlp_dat_gmv_csf[mlp_dat_gmv_csf$Experiment == idx & mlp_dat_gmv_csf$Dataset == "NACC", ],
        file = paste0(DATA_DIR, "mlp_exp_gmv_csf_exp", idx, "_NACC.csv"),
        quote = FALSE,
        row.names = FALSE
    )
}



################################################################
####### Start Weibull
################################################################

weibull <- read.csv(paste0(DATA_DIR, "weibull_predictions.csv"))
weibull$Dataset <- as.factor(weibull$Dataset)
weibull$RID <- as.character(weibull$RID)
weibull$Experiment <- as.factor(weibull$Experiment)

# extract all the predictions
preds <- NULL
for (i in 0:9) {
    preds <- cbind(preds, weibull[weibull$Bins == 12 * i, "Predictions"])
}

exp_idx <- c()
experiment <- summary(weibull$Experiment) / 10

for (i in 0:4) {
    exp_idx <- c(exp_idx, rep(i, experiment[[i + 1]]))
}

# hits/observed base truth dataframe
hits <- data.frame(
    "RID" = weibull[seq(1, nrow(weibull), 10), "RID"],
    "Experiment" = exp_idx,
    "Dataset" = weibull[seq(1, nrow(weibull), 10), "Dataset"],
    "hit" = weibull[seq(1, nrow(weibull), 10), "Progresses"],
    "observed" = weibull[seq(1, nrow(weibull), 10), "Time"]
)

weibull <- data.frame(
    "RID" = weibull[seq(1, nrow(weibull), 10), "RID"],
    "Experiment" = exp_idx,
    "Dataset" = weibull[seq(1, nrow(weibull), 10), "Dataset"]
)

for (e in seq_len(ncol(preds))) {
    weibull[, paste0("pred.prob.", (e - 1) * 12)] <- preds[, e]
}

# write experiemnt-wise
for (idx in 0:4) {

    adni_idx <- weibull[weibull$Dataset == "ADNI", ]
    adni_idx <- adni_idx[match(adni_order, adni_idx$RID), ]
    adni_idx <- adni_idx[adni_idx$Experiment == idx, ]

    nacc_idx <- weibull[weibull$Experiment == idx & weibull$Dataset == "NACC", ]
    nacc_idx <- nacc_idx[match(nacc_order, nacc_idx$RID), ]

    hits_adni <- hits[hits$Dataset == "ADNI", ]
    hits_adni <- hits_adni[match(adni_order, hits_adni$RID), ]
    hits_adni <- hits_adni[hits_adni$Experiment == idx, ]

    hits_nacc <- hits[hits$Experiment == idx & hits$Dataset == "NACC", ]
    hits_nacc <- hits_nacc[match(nacc_order, hits_nacc$RID), ]


    write.csv(adni_idx,
        file = paste0(DATA_DIR, "weibull_exp", idx, "_ADNI_test.csv"),
        quote = FALSE,
        row.names = FALSE
    )

    write.csv(nacc_idx,
        file = paste0(DATA_DIR, "weibull_exp", idx, "_NACC.csv"),
        quote = FALSE,
        row.names = FALSE
    )

    # Create hit/observed base truth for each subject
    write.csv(hits_adni,
        file = paste0(DATA_DIR, "truth_exp", idx, "_ADNI_test.csv"),
        quote = FALSE,
        row.names = FALSE
    )

    write.csv(hits_nacc,
        file = paste0(DATA_DIR, "truth_exp", idx, "_NACC.csv"),
        quote = FALSE,
        row.names = FALSE
    )
}

################################################################
####### Start CPH
################################################################

CPH_DIR <- paste0(DATA_DIR, "../cph_predicts/") # nolint

clean_cph <- function(df) {
    df <- subset(df, select = -c(X, observe, hit)) # nolint
    colnames(df) <- paste0(
        "pred.prob.",
        stringr::str_extract(colnames(df), "\\d+")
    )
    colnames(df)[13] <- "RID"
    df <- df[, c(13, 1:12)]
    df <- df[, -2]
    df$RID <- as.character(df$RID)
    df$RID <- stringr::str_pad(df$RID, width = 4, side = "left", pad = "0")
    df$RID <- as.factor(df$RID)
    df$pred.prob.0 <- "1"
    df
}

for (i in 0:4) {
    cph <- read.csv(paste0(CPH_DIR, "cph_exp", i, "_ADNI_test.csv"))

    cph <- clean_cph(cph)

    mlp_adni <- mlp_dat[mlp_dat$Experiment == i & mlp_dat$Dataset == "ADNI", ]

    write.csv(cph[match(mlp_adni$RID, cph$RID), ],
        file = paste0(DATA_DIR, "cph_exp", i, "_ADNI_test.csv"),
        quote = FALSE,
        row.names = FALSE
    )

    cph <- read.csv(paste0(CPH_DIR, "cph_exp", i, "_NACC.csv"))
    cph <- clean_cph(cph)

    write.csv(cph[match(nacc_order, cph$RID), ],
        file = paste0(DATA_DIR, "cph_exp", i, "_NACC.csv"),
        quote = FALSE,
        row.names = FALSE
    )
}


################################################################
####### Start CNN
################################################################

CNN_DIR <- paste0(DATA_DIR, "../predicts_cnn/predicts/") # nolint

clean_cnn <- function(df) {
    df <- subset(df, select = -c(X, observe, hit)) # nolint
    colnames(df) <- paste0(
        "pred.prob.",
        stringr::str_extract(colnames(df), "\\d+")
    )
    colnames(df)[1] <- "RID"
    df$RID <- as.character(df$RID)
    df$RID <- stringr::str_pad(df$RID, width = 4, side = "left", pad = "0")
    df$RID <- as.factor(df$RID)
    df$pred.prob.0 <- "1"
    df <- df[, c(1, ncol(df), 2:(ncol(df) - 1))]
    df
}

for (i in 0:4) {
    cnn <- read.csv(paste0(CNN_DIR, "cnn_mri_surv_tra_2_F1_exp", i, "_ADNI_test.csv"))
    cnn <- clean_cnn(cnn)

    mlp_adni <- mlp_dat[mlp_dat$Experiment == i & mlp_dat$Dataset == "ADNI", ]

    write.csv(cnn[match(mlp_adni$RID, cnn$RID), ],
        file = paste0(DATA_DIR, "cnn_exp", i, "_ADNI_test.csv"),
        quote = FALSE,
        row.names = FALSE
    )

    cnn <- read.csv(paste0(CNN_DIR, "cnn_mri_surv_tra_2_F1_exp", i, "_NACC.csv"))
    cnn <- clean_cnn(cnn)

    write.csv(cnn[match(nacc_order, cnn$RID), ],
        file = paste0(DATA_DIR, "cnn_exp", i, "_NACC.csv"),
        quote = FALSE,
        row.names = FALSE
    )
}

