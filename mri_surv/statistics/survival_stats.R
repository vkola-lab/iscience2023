### DEPRECATED FOR NOW, but keep this file
library(ggplot2)
library(SurvMetrics)
library(caret)
library(survival)
library(pec)
library(tidyverse)

clean_col <- function(data, dtcol) {
    data[, dtcol] <- str_remove_all(data[, dtcol], "\\[")
    data[, dtcol] <- str_remove_all(data[, dtcol], "\\]")


    split_str <- if (dtcol == "pred_prob") {
        str_split(data[, dtcol], ", ")
    } else {
        str_split(data[, dtcol], " ")
    }
    split_str <- lapply(split_str, function(x) {
        x[!x == ""]
    })
    split_str <- sapply(split_str, function(x) {
        length(x) <- 3
        return(x)
    })

    pred_matrix <- matrix(as.double(split_str), ncol = ncol(split_str))
    colname <- str_split(dtcol, "_")[[1]]

    data[, paste(colname[1], colname[2], 24, sep = ".")] <- pred_matrix[1, ]
    data[, paste(colname[1], colname[2], 48, sep = ".")] <- pred_matrix[2, ]
    data[, paste(colname[1], colname[2], 108, sep = ".")] <- pred_matrix[3, ]
    data[, dtcol] <- NULL
    data
}

clean_csv <- function(data) {
    data$X <- NULL
    data$rid <- str_extract(data$rid, pattern = "\\d+")

    data <- clean_col(data, "pred_raw")
    data <- clean_col(data, "pred_prob")
    data
}

DATA_DIR <- "/data2/MRI_PET_DATA/predicts/"
MODELS <- c("vit_Linear_exp") # model file names
MODEL_NAMES <- c("VIT") # actual model names

cindex_df <- 0
bs_df <- 0

for (modelIdx in seq_along(MODELS)) { # for each model
    for (i in 0:4) { # for each fold in each model
        dt_adni <- read.csv(paste0(
            DATA_DIR,
            MODELS[modelIdx],
            i,
            "_ADNI_test.csv"
        ))
        dt_adni <- clean_csv(dt_adni)

        dt_nacc <- read.csv(paste0(
            DATA_DIR,
            MODELS[modelIdx],
            i,
            "_NACC.csv"
        ))
        dt_nacc <- clean_csv(dt_nacc)

        adni_ci <- c()
        nacc_ci <- c()
        adni_bs <- c()
        nacc_bs <- c()


        for (time in c(24, 48, 108)) {
            ### C-INDEX
            adni_ci <- c(
                adni_ci,
                Cindex(
                    Surv(dt_adni$observe, dt_adni$hit, type = "right"),
                    dt_adni[, paste0("pred.prob.", time)]
                )
            )

            nacc_ci <- c(
                nacc_ci,
                Cindex(
                    Surv(dt_nacc$observe, dt_nacc$hit, type = "right"),
                    dt_nacc[, paste0("pred.prob.", time)]
                )
            )
            ### END C-INDEX

            ### BRIER SCORE
            adni_bs <- c(
                adni_bs,
                Brier(Surv(dt_adni$observe, dt_adni$hit, type = "right"),
                    pre_sp = dt_adni[, paste0("pred.prob.", time)],
                    t_star = time
                )
            )

            nacc_bs <- c(
                nacc_bs,
                Brier(Surv(dt_nacc$observe, dt_nacc$hit, type = "right"),
                    pre_sp = dt_nacc[, paste0("pred.prob.", time)],
                    t_star = time
                )
            )
            ### END BRIER SCORE
        }


        ### C-INDEX
        adni_ci <- c(unname(adni_ci), "ADNI", paste0("Fold.", i + 1))
        nacc_ci <- c(unname(nacc_ci), "NACC", paste0("Fold.", i + 1))
        cindex_df <- rbind(cindex_df, adni_ci, nacc_ci)
        ### END C-Index

        ### BRIER SCORE
        adni_bs <- c(unname(adni_bs), "ADNI", paste0("Fold.", i + 1))
        nacc_bs <- c(unname(nacc_bs), "NACC", paste0("Fold.", i + 1))
        bs_df <- rbind(bs_df, adni_bs, nacc_bs)
        ### END BRIER SCORE
    }

    colnames(cindex_df) <- c("m24", "m48", "m108", "dataset", "fold")
    colnames(bs_df) <- c("m24", "m48", "m108", "dataset", "fold")

    rownames(cindex_df) <- NULL
    rownames(bs_df) <- NULL
    cindex_df <- cindex_df[-1, ]
    bs_df <- bs_df[-1, ]

    cindex_df <- data.frame(cindex_df, stringsAsFactors = TRUE)
    bs_df <- data.frame(bs_df, stringsAsFactors = TRUE)
}

MODELS <- c("mlp", "cph")
MODEL_NAMES <- c("MLP", "CPH")
