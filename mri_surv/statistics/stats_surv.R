library(ggplot2)
library(SurvMetrics)
library(caret)
library(survival)
library(pec)
library(tidyverse)

DATA_DIR <- "/data2/MRI_PET_DATA/predicts/"
MODELS <- c("mlp", "cnn") # model file names
OTHER_MODELS <- c("cph", "weibull")

MODEL_TIMES <- list(
    "mlp" = c(0, 24, 48, 108),
    "cnn" = c(0, 24, 48, 108)
)


######################################
######## Our models
######################################

for (modelIdx in seq_along(MODELS)) { # for each model
    for (i in 0:4) { # for each fold in each model
        tr_adni <- read.csv(paste0(DATA_DIR, "truth_exp", i, "_ADNI_test.csv"))
        tr_adni$RID <- stringr::str_pad(tr_adni$RID, width = 4, side = "left", pad = "0")
        dt_adni <- read.csv(paste0(
            DATA_DIR,
            MODELS[modelIdx],
            "_exp",
            i,
            "_ADNI_test.csv"
        ))

        tr_nacc <- read.csv(paste0(DATA_DIR, "truth_exp", i, "_NACC.csv"))
        tr_nacc$RID <- stringr::str_pad(tr_nacc$RID, width = 4, side = "left", pad = "0")
        dt_nacc <- read.csv(paste0(
            DATA_DIR,
            MODELS[modelIdx],
            "_exp",
            i,
            "_NACC.csv"
        ))

        print(toupper(MODELS[modelIdx]))

        surv_obj_adni <- Surv(tr_adni$observed, tr_adni$hit)
        surv_obj_nacc <- Surv(tr_nacc$observed, tr_nacc$hit)

        #############
        ### Add whatever things we want to calculate down below in the for loop
        #############

        for (time in MODEL_TIMES[[MODELS[modelIdx]]]) {
            ci_adni <- Cindex(surv_obj_adni, dt_adni[, paste0("pred.prob.", time)])
            ci_nacc <- Cindex(surv_obj_nacc, dt_nacc[, paste0("pred.prob.", time)])

            # print(paste0("CI ADNI ", "Fold ", i, ": ", ci_adni))
            # print(paste0("CI NACC ", "Fold ", i, ": ", ci_nacc))
        }

        ibsrange <- MODEL_TIMES[[MODELS[modelIdx]]]
        ibsrange[1] <- 0.00001

        ibs_adni <- IBS(surv_obj_adni, dt_adni[, str_detect(colnames(dt_adni), "prob")], ibsrange)
        ibs_nacc <- IBS(surv_obj_nacc, dt_nacc[, str_detect(colnames(dt_nacc), "prob")], ibsrange)
        print(paste0("IBS ADNI ", "Fold ", i, ": ", ibs_adni))
        print(paste0("IBS NACC ", "Fold ", i, ": ", ibs_nacc))
    }
}

######################################
######## Their models
######################################
