################################################################################
# Generate R reference outputs for Europe Summer 2022 validation
# Saves intermediate results so test_europe_2022.py can compare against them.
################################################################################

suppressMessages(library(lubridate))
suppressMessages(library(ISOweek))
suppressMessages(library(dlnm))
suppressMessages(library(splines))
suppressMessages(library(mixmeta))
suppressMessages(library(tsModel))
suppressMessages(library(MASS))

DATA_DIR  <- "/Users/adessler/Desktop/europe_summer_2022_heat-main"
OUT_DIR   <- "europe_ref"
dir.create(OUT_DIR, showWarnings = FALSE)
dir.create(file.path(OUT_DIR, "rr_curves"), showWarnings = FALSE)

################################################################################
# Parameters (identical to code.R)
################################################################################

DATE1_CALI <- as.Date("2015-01-01")
DATE2_CALI <- as.Date("2019-12-26")
DATE1_PRED <- as.Date("2015-01-01")
DATE2_PRED <- as.Date("2022-11-03")
DATE1_SU22 <- as.Date("2022-06-02")
DATE2_SU22 <- as.Date("2022-09-01")

VAR_FUN  <- "ns"; VAR_DEG <- NA
VAR_PRC  <- c(10, 50, 90) / 100
MIN_LAG  <- 0; MAX_LAG <- 3
DF_SEAS  <- 8
PRED_PRC <- sort(unique(c(seq(0.0,   1.0, 0.1),
                           seq(1.5,   5.0, 0.5),
                           seq(6.0,  94.0, 1.0),
                           seq(95.0, 98.5, 0.5),
                           seq(99.0,100.0, 0.1)) / 100))
MIN_PMMT <-   5 / 100
MAX_PMMT <- 100 / 100

################################################################################
# Data Preparation
################################################################################

DATATABLE <- read.csv(file.path(DATA_DIR, "data.csv"))
DATATABLE$date <- ISOweek2date(paste0(DATATABLE$year, "-W",
                                       sprintf("%02d", DATATABLE$woy), "-", 4))
METATABLE <- read.csv(file.path(DATA_DIR, "metadata.csv"))

vREG <- METATABLE$location
nREG <- length(vREG)

DATATABLE_CALI <- DATATABLE[which(DATE1_CALI <= DATATABLE$date & DATATABLE$date <= DATE2_CALI + 7 * MAX_LAG), ]
DATATABLE_PRED <- DATATABLE[which(DATE1_PRED <= DATATABLE$date & DATATABLE$date <= DATE2_PRED + 7 * MAX_LAG), ]

DATALIST_CALI <- lapply(vREG, function(x) DATATABLE_CALI[DATATABLE_CALI$location == x, ])
names(DATALIST_CALI) <- vREG
DATALIST_PRED <- lapply(vREG, function(x) DATATABLE_PRED[DATATABLE_PRED$location == x, ])
names(DATALIST_PRED) <- vREG

for (iREG in seq_len(nREG)) {
  DATALIST_CALI[[iREG]]$wop <- seq_len(nrow(DATALIST_CALI[[iREG]]))
}

################################################################################
# Stage 1: Location-Specific GLMs
################################################################################

COEF_MODEL <- matrix(NA, nREG, length(VAR_PRC) + 1, dimnames = list(vREG))
VCOV_MODEL <- vector("list", nREG); names(VCOV_MODEL) <- vREG
MMT_REG_STAGE1 <- numeric(nREG); names(MMT_REG_STAGE1) <- vREG

for (iREG in seq_len(nREG)) {
  cat("Stage 1 region", iREG, ":", vREG[iREG], "\n")

  CROSS_BASIS <- crossbasis(
    DATALIST_CALI[[iREG]]$temp,
    c(MIN_LAG, MAX_LAG),
    argvar = list(fun = VAR_FUN,
                  knots = quantile(DATALIST_CALI[[iREG]]$temp, VAR_PRC, na.rm = TRUE),
                  Boundary.knots = range(DATALIST_CALI[[iREG]]$temp, na.rm = TRUE)),
    arglag = list(fun = "integer")
  )

  GLM_MODEL <- glm(
    formula = mort ~ ns(wop, df = round(DF_SEAS * length(wop) * 7 / 365.25)) + CROSS_BASIS,
    data    = DATALIST_CALI[[iREG]],
    family  = quasipoisson,
    na.action = "na.exclude"
  )

  suppressMessages(
    CROSS_PRED_NOMETA <- crosspred(
      CROSS_BASIS, GLM_MODEL,
      at = quantile(DATALIST_CALI[[iREG]]$temp, PRED_PRC, na.rm = TRUE)
    )
  )

  MMT <- CROSS_PRED_NOMETA$predvar[
    which(PRED_PRC == MIN_PMMT) - 1 +
    which.min(CROSS_PRED_NOMETA$allRRfit[which(PRED_PRC == MIN_PMMT):which(PRED_PRC == MAX_PMMT)])
  ]
  MMT_REG_STAGE1[iREG] <- MMT

  REDUCED <- crossreduce(CROSS_BASIS, GLM_MODEL, cen = MMT)
  COEF_MODEL[iREG, ] <- coef(REDUCED)
  VCOV_MODEL[[iREG]] <- vcov(REDUCED)
}

# Save Stage 1 outputs
write.csv(COEF_MODEL, file.path(OUT_DIR, "coef_model.csv"))
saveRDS(VCOV_MODEL,   file.path(OUT_DIR, "vcov_model.rds"))
write.csv(data.frame(location = vREG, mmt = MMT_REG_STAGE1),
          file.path(OUT_DIR, "mmt_stage1.csv"), row.names = FALSE)

################################################################################
# Stage 2: Meta-Analysis and BLUPs
################################################################################

TEMP_AVG <- sapply(DATALIST_CALI, function(x) mean(x$temp, na.rm = TRUE))
TEMP_IQR <- sapply(DATALIST_CALI, function(x)  IQR(x$temp, na.rm = TRUE))

MULTIVAR <- mixmeta(
  COEF_MODEL ~ TEMP_AVG + TEMP_IQR,
  VCOV_MODEL,
  data    = data.frame(vREG = vREG),
  control = list(showiter = FALSE, igls.inititer = 10),
  method  = "reml"
)

BLUP <- blup(MULTIVAR, vcov = TRUE)

# Save BLUPs
blup_coef_mat <- do.call(rbind, lapply(BLUP, function(b) b$blup))
rownames(blup_coef_mat) <- vREG
write.csv(blup_coef_mat, file.path(OUT_DIR, "blup_coefs.csv"))
saveRDS(lapply(BLUP, function(b) b$vcov), file.path(OUT_DIR, "blup_vcov.rds"))

################################################################################
# Stage 3: Predictions (RR curves) and MMT after meta-analysis
################################################################################

MMT_REG <- numeric(nREG); names(MMT_REG) <- vREG

for (iREG in seq_len(nREG)) {
  BASIS_VAR <- onebasis(
    quantile(DATALIST_CALI[[iREG]]$temp, PRED_PRC, na.rm = TRUE),
    fun    = VAR_FUN,
    knots  = quantile(DATALIST_CALI[[iREG]]$temp, VAR_PRC, na.rm = TRUE),
    Boundary.knots = range(DATALIST_CALI[[iREG]]$temp, na.rm = TRUE)
  )
  PRC_VAR <- quantile(DATALIST_CALI[[iREG]]$temp, PRED_PRC, na.rm = TRUE)

  suppressMessages(
    PRED_MORT <- crosspred(BASIS_VAR,
                           coef = BLUP[[iREG]]$blup,
                           vcov = BLUP[[iREG]]$vcov,
                           model.link = "log",
                           at = PRC_VAR)
  )

  MMT_REG[iREG] <- PRC_VAR[
    which(PRED_PRC == MIN_PMMT) - 1 +
    which.min(PRED_MORT$allRRfit[which(PRED_PRC == MIN_PMMT):which(PRED_PRC == MAX_PMMT)])
  ]

  suppressMessages(
    CROSS_PRED_META <- crosspred(BASIS_VAR,
                                 coef = BLUP[[iREG]]$blup,
                                 vcov = BLUP[[iREG]]$vcov,
                                 model.link = "log",
                                 at = PRC_VAR,
                                 cen = MMT_REG[iREG])
  )

  write.csv(
    data.frame(
      temperature = CROSS_PRED_META$predvar,
      allRRfit    = CROSS_PRED_META$allRRfit,
      allRRlow    = CROSS_PRED_META$allRRlow,
      allRRhigh   = CROSS_PRED_META$allRRhigh
    ),
    file.path(OUT_DIR, "rr_curves", paste0("rr_", vREG[iREG], ".csv")),
    row.names = FALSE
  )
}

write.csv(data.frame(location = vREG, mmt = MMT_REG),
          file.path(OUT_DIR, "mmt_reg.csv"), row.names = FALSE)

################################################################################
# Stage 4: Attribution (Summer 2022, Total Heat / Total Cold)
################################################################################

nSIM <- 1000L
an_rows <- vector("list", nREG)

for (iREG in seq_len(nREG)) {
  cat("Attribution region", iREG, ":", vREG[iREG], "\n")

  BASIS_VAR <- onebasis(
    DATALIST_PRED[[iREG]]$temp,
    fun    = VAR_FUN,
    knots  = quantile(DATALIST_CALI[[iREG]]$temp, VAR_PRC, na.rm = TRUE),
    Boundary.knots = range(DATALIST_CALI[[iREG]]$temp, na.rm = TRUE)
  )
  BASIS_MMT <- onebasis(
    MMT_REG[iREG],
    fun    = VAR_FUN,
    knots  = quantile(DATALIST_CALI[[iREG]]$temp, VAR_PRC, na.rm = TRUE),
    Boundary.knots = range(DATALIST_CALI[[iREG]]$temp, na.rm = TRUE)
  )
  BASIS_CEN <- scale(BASIS_VAR, center = BASIS_MMT, scale = FALSE)

  LAGGED_MORT_MATRIX <- Lag(DATALIST_PRED[[iREG]]$mort, -MIN_LAG:-MAX_LAG)
  LAGGED_MORT_VECTOR <- rowMeans(LAGGED_MORT_MATRIX)

  ATT_NUM_TS_REF <- (1 - exp(-BASIS_CEN %*% BLUP[[iREG]]$blup)) * LAGGED_MORT_VECTOR

  set.seed(5634654)
  COEF_SIM <- t(mvrnorm(nSIM, BLUP[[iREG]]$blup, BLUP[[iREG]]$vcov))
  ATT_NUM_TS_SIM <- (1 - exp(-BASIS_CEN %*% COEF_SIM)) * LAGGED_MORT_VECTOR

  # Summer 2022 indices (excluding final MAX_LAG weeks)
  n_pred <- nrow(DATALIST_PRED[[iREG]]) - MAX_LAG
  pred_dates <- DATALIST_PRED[[iREG]]$date[seq_len(n_pred)]
  vTIM_SU22  <- which(DATE1_SU22 <= pred_dates & pred_dates <= DATE2_SU22)

  tmin_val <- min(DATALIST_PRED[[iREG]]$temp, na.rm = TRUE)
  p025_val <- quantile(DATALIST_PRED[[iREG]]$temp, 0.025, na.rm = TRUE)
  p975_val <- quantile(DATALIST_PRED[[iREG]]$temp, 0.975, na.rm = TRUE)
  tmax_val <- max(DATALIST_PRED[[iREG]]$temp, na.rm = TRUE)

  compute_an <- function(vTIM, temp_range_name) {
    if (temp_range_name == "Total") {
      vRNG_THRES <- seq_along(vTIM)
    } else if (temp_range_name == "Total Cold") {
      vRNG_THRES <- which(DATALIST_PRED[[iREG]]$temp[vTIM] <  MMT_REG[iREG])
    } else if (temp_range_name == "Total Heat") {
      vRNG_THRES <- which(DATALIST_PRED[[iREG]]$temp[vTIM] >  MMT_REG[iREG])
    } else {
      stop("Unknown range")
    }
    if (length(vRNG_THRES) == 0 || sum(LAGGED_MORT_VECTOR[vTIM]) == 0) return(c(0, 0, 0))
    correction <- sum(rowMeans(LAGGED_MORT_MATRIX[vTIM, , drop = FALSE], na.rm = TRUE), na.rm = TRUE) /
                  sum(LAGGED_MORT_VECTOR[vTIM], na.rm = TRUE)
    an_val  <- sum(ATT_NUM_TS_REF[vTIM[vRNG_THRES]], na.rm = TRUE) * correction
    an_sims <- colSums(ATT_NUM_TS_SIM[vTIM[vRNG_THRES], , drop = FALSE], na.rm = TRUE) * correction
    c(an_val, quantile(an_sims, c(0.025, 0.975)))
  }

  for (rng in c("Total", "Total Cold", "Total Heat")) {
    res <- compute_an(vTIM_SU22, rng)
    an_rows[[iREG]] <- rbind(an_rows[[iREG]],
      data.frame(location = vREG[iREG], period = "Summer 2022",
                 range = rng, att_val = res[1], att_low = res[2], att_upp = res[3]))
  }
}

an_df <- do.call(rbind, an_rows)
write.csv(an_df, file.path(OUT_DIR, "an_summer2022.csv"), row.names = FALSE)

cat("\nAll reference data saved to", OUT_DIR, "\n")
