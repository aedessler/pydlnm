#!/usr/bin/env Rscript
# Regenerate R reference data (BLUPs + per-region RR curves)
# Outputs saved to reference_data/ at repo root.

library(dlnm); library(splines); library(mvmeta)

DATA_PATH <- "2015_gasparrini_Lancet_Rcodedata-master/regEngWales.csv"
OUT_DIR   <- "reference_data"
dir.create(OUT_DIR, showWarnings=FALSE)

# ── Parameters (matching 00.prepdata.R / 01.firststage.R) ─────────────────────
varfun   <- "bs"; vardegree <- 2; varper <- c(10,75,90)
lag <- 21; lagnk <- 3; dfseas <- 8

data <- read.csv(DATA_PATH, row.names=1)
data$date <- as.Date(data$date)
regions   <- sort(unique(data$regnames))

coef_list <- vector("list", length(regions))
vcov_list <- vector("list", length(regions))
names(coef_list) <- names(vcov_list) <- regions

dlist <- list()
for (reg in regions) {
  d <- data[data$regnames == reg, ]
  d <- d[order(d$date), ]
  dlist[[reg]] <- d
}

cat("Running first-stage GLMs...\n")
for (reg in regions) {
  d      <- dlist[[reg]]
  varknots <- quantile(d$tmean, varper/100, na.rm=TRUE)
  lagknots <- logknots(c(0, lag), nk=lagnk)

  cb <- crossbasis(d$tmean, lag=lag,
    argvar=list(fun=varfun, knots=varknots, degree=vardegree),
    arglag=list(fun="ns",   knots=lagknots))

  nyears <- length(unique(format(d$date, "%Y")))
  model  <- glm(death ~ cb + dow + ns(date, df=dfseas*nyears),
                data=d, family=quasipoisson, na.action=na.exclude)

  cen <- mean(d$tmean, na.rm=TRUE)
  red <- crossreduce(cb, model, cen=cen)
  coef_list[[reg]] <- coef(red)
  vcov_list[[reg]] <- vcov(red)
  cat("  fitted", reg, "\n")
}

saveRDS(do.call(rbind, coef_list), file.path(OUT_DIR, "coefficients.rds"))
saveRDS(vcov_list, file.path(OUT_DIR, "vcov_matrices.rds"))

# ── Second stage ───────────────────────────────────────────────────────────────
cat("Running MVMeta...\n")
avgtmean   <- sapply(dlist, function(x) mean(x$tmean, na.rm=TRUE))
rangetmean <- sapply(dlist, function(x) diff(range(x$tmean, na.rm=TRUE)))

cities <- data.frame(avgtmean=avgtmean, rangetmean=rangetmean,
                     row.names=regions)

coef_mat <- do.call(rbind, coef_list)

mv   <- mvmeta(coef_mat ~ avgtmean + rangetmean, vcov_list,
               data=cities, control=list(showiter=FALSE))
blup_res <- blup(mv, vcov=TRUE)
saveRDS(blup_res, file.path(OUT_DIR, "blup_results.rds"))

# ── Per-region RR curves ───────────────────────────────────────────────────────
cat("Generating RR curves...\n")
for (i in seq_along(regions)) {
  reg <- regions[i]
  d   <- dlist[[reg]]
  varknots <- quantile(d$tmean, varper/100, na.rm=TRUE)
  lagknots <- logknots(c(0, lag), nk=lagnk)

  predvar  <- seq(min(d$tmean, na.rm=TRUE), max(d$tmean, na.rm=TRUE), by=0.1)
  argvar   <- list(x=predvar, fun=varfun, knots=varknots, degree=vardegree,
                   Boundary.knots=range(d$tmean, na.rm=TRUE))
  bvar     <- do.call(onebasis, argvar)

  # Find MMT
  rr_vals  <- exp(bvar %*% blup_res[[i]]$blup)
  mmt      <- predvar[which.min(rr_vals)]

  cb_pred  <- crossbasis(d$tmean, lag=lag,
    argvar=list(fun=varfun, knots=varknots, degree=vardegree),
    arglag=list(fun="ns",   knots=lagknots))

  nyears  <- length(unique(format(d$date, "%Y")))
  model_i <- glm(death ~ cb_pred + dow + ns(date, df=dfseas*nyears),
                 data=d, family=quasipoisson, na.action=na.exclude)

  pred <- crosspred(cb_pred, model_i, coef=blup_res[[i]]$blup,
                    vcov=blup_res[[i]]$vcov, model.link="log",
                    at=predvar, cen=mmt)

  csv_stem <- gsub(" & ", "___", gsub(" ", "_", reg))
  write.csv(data.frame(
    temperature = predvar,
    rr_fit      = pred$allRRfit,
    rr_low      = pred$allRRlow,
    rr_high     = pred$allRRhigh,
    mmt         = mmt
  ), file.path(OUT_DIR, paste0("rr_curve_", csv_stem, ".csv")),
  row.names=FALSE)
  cat("  saved", reg, "\n")
}

cat("\nDone. Reference data in", OUT_DIR, "\n")
