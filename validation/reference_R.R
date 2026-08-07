## Reference run: the original R implementation on validation/simdata.csv.
## Point estimates from this script are what the Python and Stata ports are
## checked against.  Bootstrap standard errors are not comparable across
## languages -- different RNGs draw different resamples -- so nboots is kept
## small here and only the coefficients are written out.

fn_dir <- Sys.getenv("COPREG_FN_DIR")
if (!nzchar(fn_dir))
  stop("Set COPREG_FN_DIR to the directory holding Copreg_*.R")

for (f in c("Copreg_core.R", "Copreg_pg.R", "Copreg_2scope.R", "Copreg_ima.R",
            "Copreg_jams.R", "Copreg_bmw.R"))
  source(file.path(fn_dir, f))

d <- read.csv("simdata.csv")
d$g <- as.factor(d$g)

set.seed(1)
runs <- list(
  pg      = CopRegPG(y ~ P | x + w + g, data = d, nboots = 5, verbose = FALSE),
  s2cope  = CopReg2sCOPE(y ~ P | x + w + g, data = d, nboots = 5, verbose = FALSE),
  ima     = CopRegIMA(y ~ P | x + w + g, data = d, nboots = 5, verbose = FALSE),
  bmw     = CopRegBMW(y ~ P | x + w + g, data = d, nboots = 5, verbose = FALSE),
  jams    = CopRegJAMS(y ~ P | x + w + g, data = d, nboots = 5, verbose = FALSE),
  jams_f  = CopRegJAMS(y ~ P | x + w + g, data = d, conditional = FALSE,
                       nboots = 5, verbose = FALSE),
  pg_ecdf = CopRegPG(y ~ P | x + w, data = d, cdf = "ecdf.adj", nboots = 5,
                     verbose = FALSE),
  pg_rank = CopRegPG(y ~ P | x + w, data = d, cdf = "rank.n", nboots = 5,
                     verbose = FALSE),
  pg_plug = CopRegPG(y ~ P | x + w, data = d, cdf = "kde.plugin", nboots = 5,
                     verbose = FALSE),
  pg_int  = CopRegPG(y ~ P | x + w + x:w, data = d, nboots = 5, verbose = FALSE)
)

out <- do.call(rbind, lapply(names(runs), function(nm) {
  m <- runs[[nm]]
  data.frame(model = nm, term = names(coef(m)), estimate = unname(coef(m)),
             stringsAsFactors = FALSE)
}))

rho <- do.call(rbind, lapply(names(runs), function(nm) {
  m <- runs[[nm]]
  data.frame(model = nm, term = paste0("rho.", names(m$rho)),
             estimate = unname(m$rho), stringsAsFactors = FALSE)
}))

write.csv(rbind(out, rho), "reference_R.csv", row.names = FALSE)

## the CDF estimators on their own, for a column-by-column check
u <- sapply(c("kde.silverman", "kde.plugin", "ecdf.fixed", "ecdf.adj",
              "rank.n", "rank.n1"),
            function(k) .cdf_estimate(d$P, k, "max"))
write.csv(as.data.frame(u), "reference_R_cdf.csv", row.names = FALSE)

cat("written\n")
