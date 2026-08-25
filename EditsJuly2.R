# ==============================================================================
# GAME 5 — CLEAN PIPELINE (5-level bins; O2/H2O/Tmin/Tmax; Tavg for starting mix)
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(purrr)
  library(grid)
  library(car)
  library(effectsize)
  library(nnet)
})

# ---------------------------
# 0) CONFIG
# ---------------------------
STRATS <- c("Viviparity","Oviparity","O.C")
STRAT_COLORS <- c("Viviparity"="#D55E00", "Oviparity"="#0072B2", "O.C"="#009E73")
lev5 <- c("EL","L","M","H","EH")

OUT_DIR <- "thesis_outputs"
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

CACHE_FILE <- file.path(OUT_DIR, "results_cache.rds")

# ---------------------------
# 1) HELPERS
# ---------------------------
quantile_levels5 <- function(x, labels = lev5) {
  qs <- stats::quantile(x, probs = seq(0, 1, by = 0.2), na.rm = TRUE, type = 7)
  qs_unique <- unique(qs)
  if (length(qs_unique) < 6) {
    r <- rank(x, na.last = "keep", ties.method = "average")
    return(cut(r,
               breaks = stats::quantile(r, probs = seq(0, 1, by = 0.2), na.rm = TRUE),
               include.lowest = TRUE, labels = labels
    ))
  }
  cut(x, breaks = qs, include.lowest = TRUE, labels = labels)
}

lvl_num <- function(x) {
  out <- match(x, lev5) - 3
  if (any(is.na(out))) stop("Unknown level(s): ", paste(unique(x[is.na(out)]), collapse=", "))
  out
}

env_stress <- function(O2, H2O, Tmin, Tmax) {
  abs(lvl_num(O2)) + abs(lvl_num(H2O)) + abs(lvl_num(Tmin)) + abs(lvl_num(Tmax))  # 0..8
}

initial_from_Tavg <- function(Tavg_level) {
  switch(Tavg_level,
         "EL" = c(0.70, 0.15, 0.15),
         "L"  = c(0.55, 0.25, 0.20),
         "M"  = c(1/3, 1/3, 1/3),
         "H"  = c(0.25, 0.55, 0.20),
         "EH" = c(0.15, 0.70, 0.15),
         c(1/3, 1/3, 1/3)
  )
}

# ---------------------------
# 2) LOAD CLIMATE DATA + BIN TO LEVELS
# ---------------------------
clim_path <- "combined_data_with_climate_and_elev_hand_mod.csv"
clim <- read.csv(clim_path, stringsAsFactors = FALSE)

stopifnot(all(c("Prec","Vapr","Elev","Tmin","Tmax","Tavg") %in% names(clim)))

# H2O index from Prec + Vapr
clim$H2O_index <- as.numeric(scale(log1p(clim$Prec))) + as.numeric(scale(clim$Vapr))
clim$H2O_level <- as.character(quantile_levels5(clim$H2O_index))

# O2 proxy from elevation (higher elev => lower effective O2)
clim$O2_proxy_index <- -as.numeric(scale(clim$Elev))
clim$O2_level <- as.character(quantile_levels5(clim$O2_proxy_index))

# Temp levels
clim$Tmin_level <- as.character(quantile_levels5(clim$Tmin))
clim$Tmax_level <- as.character(quantile_levels5(clim$Tmax))
clim$Tavg_level <- as.character(quantile_levels5(clim$Tavg))

# Observed environments (from data)
env_grid <- clim %>%
  transmute(
    O2   = O2_level,
    H2O  = H2O_level,
    Tmin = Tmin_level,
    Tmax = Tmax_level,
    Tavg = Tavg_level
  ) %>%
  distinct() %>%
  mutate(
    env4 = paste(O2, H2O, Tmin, Tmax, sep = "-"),
    env5 = paste(O2, H2O, Tmin, Tmax, Tavg, sep = "-")
  )
env_grid
#O2, H2O, Tmin, Tmax, Tavg
# env4 = O2-H2O-Tmin-Tmax (4 axes used in the game)
# env5 = env4 + Tavg (only used to choose starting mix)

cat("Observed unique env4:", n_distinct(env_grid$env4), "\n")
cat("Observed unique env5:", n_distinct(env_grid$env5), "\n")

# ---------------------------
# 3) YOUR oc_table -> FIT_PAR
# ---------------------------
oc_table_raw <- tibble::tribble(
  ~Group,                 ~Species,                                   ~O2_capacity_volpct, ~Temperature_C, ~Elevation_m_text, ~Citation,
  
  # =========================
  # Lizards — Oviparous
  # =========================
  "Lizards-Oviparous",    "Acanthodactylus erythrurus",               4.7,   NA,        NA,           "Pough 1979",
  "Lizards-Oviparous",    "Aspidoscelis inornatus",                   10.8,  "39.3",    "1646–1981",   "Dawson & Poulson 1962",
  "Lizards-Oviparous",    "Aspidoscelis sackii",                      11.6,  "39.4",    "1310–1402",   "Dawson & Poulson 1962",
  "Lizards-Oviparous",    "Aspidoscelis tigris",                      9.6,   "41.3",    "1402",        "Dawson & Poulson 1962",
  "Lizards-Oviparous",    "Cophosaurus texanus",                      7.8,   "38.2",    "1402",        "Dawson & Poulson 1962",
  "Lizards-Oviparous",    "Crotaphytus collaris",                     10.4,  "37.8",    "610",         "Dawson & Poulson 1962",
  "Lizards-Oviparous",    "Dipsosaurus dorsalis",                     9.6,   "42.1",    "183",         "Dawson & Poulson 1962",
  "Lizards-Oviparous",    "Dipsosaurus dorsalis",                     9.3,   "30-35",   NA,            "Pough 1976",
  "Lizards-Oviparous",    "Elgaria multicarinata",                    12.6,  "25.8",    "183",         "Dawson & Poulson 1962",
  "Lizards-Oviparous",    "Elgaria multicarinata",                    6.7,   "15–30",   NA,            "Pough 1976",
  "Lizards-Oviparous",    "Gambelia wislizeni",                       10.4,  "38.9",    "610",         "Dawson & Poulson 1962",
  "Lizards-Oviparous",    "Iguana iguana",                            7.9,   "20–35",   NA,            "Pough 1976",
  "Lizards-Oviparous",    "Iguana iguana",                            8.4,   "28–32",   NA,            "Tucker 1966",
  "Lizards-Oviparous",    "Iguana iguana",                            10.5,  NA,        NA,            "Wood & Moberly 1970",
  "Lizards-Oviparous",    "Paralaudakia caucasia",                    11.7,  NA,        NA,            "Verjbiinskaya 1944",
  "Lizards-Oviparous",    "Phrynosoma cornutum",                      9.3,   "37.3",    "1646",        "Dawson & Poulson 1962",
  "Lizards-Oviparous",    "Phrynosoma modestum",                      12.1,  NA,        "1402",        "Dawson & Poulson 1962",
  "Lizards-Oviparous",    "Psammodromus algirus",                     4.0,   NA,        NA,            "Pough 1979",
  "Lizards-Oviparous",    "Pseudopus apodus",                         9.8,   NA,        NA,            "Verjbiinskaya 1944",
  "Lizards-Oviparous",    "Sauromalus hispidus",                      9.7,   "25",      NA,            "Bennett 1973",
  "Lizards-Oviparous",    "Sauromalus obesus",                        9.8,   "30–35",   NA,            "Pough 1976",
  "Lizards-Oviparous",    "Sauromalus obesus",                        10.8,  NA,        NA,            "Pough 1979",
  "Lizards-Oviparous",    "Sceloporus clarki",                        8.3,   NA,        "1554",        "Dawson & Poulson 1962",
  "Lizards-Oviparous",    "Sceloporus graciosus",                     10.9,  NA,        "183",         "Dawson & Poulson 1962",
  "Lizards-Oviparous",    "Sceloporus magister",                      8.9,   "30",      NA,            "Pough 1976",
  "Lizards-Oviparous",    "Sceloporus magister",                      10.9,  NA,        NA,            "Ryerson 1949",
  "Lizards-Oviparous",    "Sceloporus occidentalis",                  9.5,   "33.8",    "183",         "Dawson & Poulson 1962",
  "Lizards-Oviparous",    "Sceloporus occidentalis",                  5.8,   "20–35",   NA,            "Pough 1976",
  "Lizards-Oviparous",    "Sceloporus orcutti",                       8.0,   "25–30",   NA,            "Pough 1976",
  "Lizards-Oviparous",    "Sceloporus virgatus",                      9.6,   "34.8",    "1554–1707",   "Dawson & Poulson 1962",
  "Lizards-Oviparous",    "Tupinambis merianae (Summer)",             13.1,  "25",      NA,            "Andrade et al. 2004",
  "Lizards-Oviparous",    "Tupinambis merianae (Winter)",             11.02, "17",      NA,            "Andrade et al. 2004",
  "Lizards-Oviparous",    "Tupinambis teguixin",                      9.5,   "35",      NA,            "Pough 1976",
  "Lizards-Oviparous",    "Uta stansburiana",                         8.7,   "35–38",   "1402",        "Dawson & Poulson 1962",
  "Lizards-Oviparous",    "Uta stansburiana",                         9.2,   "20–35",   NA,            "Pough 1976",
  "Lizards-Oviparous",    "Varanus examanthematicus",                 10.1,  NA,        NA,            "Wood et al. 1977",
  "Lizards-Oviparous",    "Varanus gouldi",                           8.0,   "25",      NA,            "Bennett 1973",
  "Lizards-Oviparous",    "Varanus niloticus",                        9.3,   "25",      NA,            "Wood & Johansen 1974",
  
  # =========================
  # Lizards — Viviparous
  # =========================
  "Lizards-Viviparous",   "Elgaria coerulea",                          6.8,   NA,        NA,          "Pough 1979",
  "Lizards-Viviparous",   "Phrynosoma douglassi",                      10.2,  "36",      "1646",      "Dawson & Poulson 1962",
  "Lizards-Viviparous",   "Sceloporus jarrovi",                        7.7,   "34.8",    "1554",      "Dawson & Poulson 1962",
  "Lizards-Viviparous",   "Sceloporus jarrovi",                        8.3,   "34.8",    "2621",      "Dawson & Poulson 1962",
  "Lizards-Viviparous",   "Sceloporus poinsetti",                      10.0,  "34.2",    "1341",      "Dawson & Poulson 1962",
  
  # =========================
  # Snakes — Oviparous
  # =========================
  "Snakes-Oviparous",     "Drymarchon corais",                         4.2,   NA,        NA,          "Pough 1979",
  "Snakes-Oviparous",     "Lampropeltis getula",                       4.5,   NA,        NA,          "Pough 1979",
  "Snakes-Oviparous",     "Lampropeltis triangulum",                   8.7,   "20–25",   NA,          "Pough 1976",
  "Snakes-Oviparous",     "Laticauda colubrina",                       11.0,  NA,        NA,          "Pough 1979",
  "Snakes-Oviparous",     "Laticauda colubrina",                       10.7,  "28",      NA,          "Pough & Lillywhite 1984",
  "Snakes-Oviparous",     "Laticauda columbrina",                      11.04, "30",      NA,          "Feder 1980",
  "Snakes-Oviparous",     "Laticauda columbrina",                      9.3,   "27",      NA,          "Seymore 1976",
  "Snakes-Oviparous",     "Pantherophis guttatus",                     4.2,   NA,        NA,          "Pough 1979",
  "Snakes-Oviparous",     "Pantherophis obsoletus",                    6.7,   NA,        NA,          "Pough 1979",
  "Snakes-Oviparous",     "Pituophis melanoleucus",                    9.3,   NA,        NA,          "Greenwald 1971",
  "Snakes-Oviparous",     "Pituophis melanoleucus",                    11.4,  "20–25",   NA,          "Pough 1976",
  "Snakes-Oviparous",     "Python molurus",                            5.7,   NA,        NA,          "Pough 1979",
  "Snakes-Oviparous",     "Python reticulatus",                        7.1,   NA,        NA,          "Pough 1979",
  "Snakes-Oviparous",     "Xenopeltis unicolor",                       10.0,  NA,        NA,          "Pough 1979",
  
  # =========================
  # Snakes — Viviparous
  # =========================
  "Snakes-Viviparous",    "Acalyptophis peronii",                      7.9,   "28",      NA,          "Pough & Lillywhite 1984",
  "Snakes-Viviparous",    "Acrocordus granulatus",                     16.2,  "30",      NA,          "Feder 1980",
  "Snakes-Viviparous",    "Acrocordus javanicus",                      9.3,   NA,        NA,          "Johansen & Lenfant 1972",
  "Snakes-Viviparous",    "Agkistrodon piscivorous (Fetal)",           7.1,   "23",      NA,          "Birchard et al. 1984a",
  "Snakes-Viviparous",    "Agkistrodon piscivorous (Maternal)",        7.6,   "23",      NA,          "Birchard et al. 1984a",
  "Snakes-Viviparous",    "Aipysurus laevis",                          12.7,  "28",      NA,          "Pough & Lillywhite 1984",
  "Snakes-Viviparous",    "Boa constrictor",                           13.5,  NA,        NA,          "Johansen & Lenfant 1972",
  "Snakes-Viviparous",    "Cerberus rhynchops",                        3.6,   "30",      NA,          "Feder 1980",
  "Snakes-Viviparous",    "Emydocephalus annulatus",                   6.53,  "30",      NA,          "Feder 1980",
  "Snakes-Viviparous",    "Emydocephalus annulatus",                   6.5,   NA,        NA,          "Pough 1979",
  "Snakes-Viviparous",    "Eryx johni",                                6.4,   NA,        NA,          "Pough 1979",
  "Snakes-Viviparous",    "Hydrophis melanocephalus",                  10.8,  "28",      NA,          "Pough & Lillywhite 1984",
  "Snakes-Viviparous",    "Nerodia rhombifer",                         7.6,   NA,        NA,          "Pough 1979",
  "Snakes-Viviparous",    "Nerodia sipedon",                           11.1,  "20–25",   NA,          "Pough 1976",
  "Snakes-Viviparous",    "Pelamis platura",                           10.2,  "28",      NA,          "Pough & Lillywhite 1984",
  "Snakes-Viviparous",    "Pseudechis porphyriacus",                   10.9,  "27",      NA,          "Seymour 1976",
  "Snakes-Viviparous",    "Thamnophis couchi",                         9.0,   NA,        NA,          "Pough 1979",
  "Snakes-Viviparous",    "Thamnophis sirtalis (Males)",               9.6,   NA,        NA,          "Birchard et al. 1984b",
  "Snakes-Viviparous",    "Thamnophis sirtalis (Nongravid females)",   9.2,   NA,        NA,          "Birchard et al. 1984b",
  "Snakes-Viviparous",    "Thamnophis sirtalis (Gravid Females)",      11.1,  NA,        NA,          "Birchard et al. 1984b",
  "Snakes-Viviparous",    "Thamnophis sirtalis (Post-partum females)", 9.7,   NA,        NA,          "Birchard et al. 1984b",
  "Snakes-Viviparous",    "Thamnophis sirtalis",                       9.5,   "10–30",   NA,          "Pough 1976",
  "Snakes-Viviparous",    "Vipera berus",                              12.3,  "5",       NA,          "Johansen & Lykkeboe 1979",
  "Snakes-Viviparous",    "Vipera berus",                              9.9,   "25",      NA,          "Johansen & Lykkeboe 1979"
)

# ---- Parse helper for numeric or range fields (temperature or elevation) ----
parse_range_num <- function(x) {
  if (is.na(x) || !nzchar(x)) return(c(NA_real_, NA_real_, NA_real_))
  xs <- stringr::str_replace_all(x, "[–—−]", "-")
  xs <- stringr::str_replace_all(xs, "(?<=\\d)-(?=\\d)", " ")
  nums <- stringr::str_extract_all(xs, "-?\\d+(?:\\.\\d+)?")[[1]]
  nums <- suppressWarnings(as.numeric(nums))
  nums <- nums[!is.na(nums)]
  if (length(nums) == 0) return(c(NA_real_, NA_real_, NA_real_))
  if (length(nums) == 1) return(c(nums[1], nums[1], nums[1]))
  a <- nums[1]; b <- nums[2]
  c(min(a, b), max(a, b), mean(c(a, b)))
}

# ---- Build parsed table ----
oc_table <- oc_table_raw %>%
  mutate(
    T_vals = map(Temperature_C, parse_range_num),
    T_min  = map_dbl(T_vals, 1),
    T_max  = map_dbl(T_vals, 2),
    T_mid  = map_dbl(T_vals, 3),
    E_vals = map(Elevation_m_text, parse_range_num),
    Elev_min = map_dbl(E_vals, 1),
    Elev_max = map_dbl(E_vals, 2),
    Elev_mid = map_dbl(E_vals, 3)
  ) %>%
  select(-T_vals, -E_vals) %>%
  mutate(
    Parity = case_when(
      str_detect(Group, "Viviparous") ~ "Viviparous",
      str_detect(Group, "Oviparous")  ~ "Oviparous",
      TRUE ~ "Unknown"
    )
  )
head(oc_table)
# A tibble: 6 × 13
#Group   Species O2_capacity_volpct Temperature_C Elevation_m_text Citation T_min T_max T_mid
#<chr>   <chr>                <dbl> <chr>         <chr>            <chr>    <dbl> <dbl> <dbl>
#  1 Lizard… Acanth…                4.7 NA            NA               Pough 1…  NA    NA    NA  
#2 Lizard… Aspido…               10.8 39.3          1646–1981        Dawson …  39.3  39.3  39.3
#3 Lizard… Aspido…               11.6 39.4          1310–1402        Dawson …  39.4  39.4  39.4
#4 Lizard… Aspido…                9.6 41.3          1402             Dawson …  41.3  41.3  41.3
#5 Lizard… Cophos…                7.8 38.2          1402             Dawson …  38.2  38.2  38.2
#6 Lizard… Crotap…               10.4 37.8          610              Dawson …  37.8  37.8  37.8

# ==============================================================================
# SUMMARIES FROM THE TABLE
# ==============================================================================

safe_mean <- function(x) if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
safe_sd   <- function(x) if (all(is.na(x))) NA_real_ else stats::sd(x, na.rm = TRUE)
safe_min  <- function(x) if (all(is.na(x))) NA_real_ else min(x,  na.rm = TRUE)
safe_max  <- function(x) if (all(is.na(x))) NA_real_ else max(x,  na.rm = TRUE)

oc_summary_by_parity <- oc_table %>%
  group_by(Parity) %>%
  summarise(
    n = n(),
    O2_mean = safe_mean(O2_capacity_volpct),
    O2_sd   = safe_sd(O2_capacity_volpct),
    O2_min  = safe_min(O2_capacity_volpct),
    O2_max  = safe_max(O2_capacity_volpct),
    
    n_T = sum(!is.na(T_mid)),
    T_mean = safe_mean(T_mid),
    T_sd   = safe_sd(T_mid),
    T_min  = safe_min(T_mid),
    T_max  = safe_max(T_mid),
    
    n_Elev = sum(!is.na(Elev_mid)),
    Elev_mean = safe_mean(Elev_mid),
    Elev_sd   = safe_sd(Elev_mid),
    Elev_min  = safe_min(Elev_mid),
    Elev_max  = safe_max(Elev_mid),
    
    .groups = "drop"
  )

print(oc_summary_by_parity)
## A tibble: 2 × 16
#Parity         n O2_mean O2_sd O2_min O2_max   n_T T_mean  T_sd T_min T_max n_Elev Elev_mean
#<chr>      <int>   <dbl> <dbl>  <dbl>  <dbl> <int>  <dbl> <dbl> <dbl> <dbl>  <int>     <dbl>
#  1 Oviparous     52    9.08  2.22    4     13.1    32   30.8  6.48    17  42.1     15     1037.
#2 Viviparous    29    9.35  2.53    3.6   16.2    18   27.1  7.17     5  36        4     1790.

# ==============================================================================
# EXPORT EMPIRICAL TABLES (now that OUT_DIR exists)
# ==============================================================================
write.csv(oc_table,
          file = file.path(OUT_DIR, "table_empirical_O2_T_elevation.csv"),
          row.names = FALSE)

write.csv(oc_summary_by_parity,
          file = file.path(OUT_DIR, "table_empirical_summary_by_parity.csv"),
          row.names = FALSE)

cat("\nSaved empirical tables to:", OUT_DIR, "\n")

# ==============================================================================
# DERIVE FIT_PAR FROM REAL NUMBERS (NO HAND-ENTERED COEFFICIENTS)
# ==============================================================================

# Helper: pooled SD for standardized differences (Cohen's d style)
pooled_sd <- function(m1, s1, n1, m2, s2, n2) {
  if (!is.finite(s1) || !is.finite(s2) || !is.finite(n1) || !is.finite(n2)) return(NA_real_)
  if (n1 < 2 || n2 < 2) return(NA_real_)
  sqrt(((n1 - 1) * s1^2 + (n2 - 1) * s2^2) / (n1 + n2 - 2))
}

clamp <- function(x, lo, hi) pmax(lo, pmin(hi, x))

derive_fit_par <- function(oc_summary_by_parity) {
  
  get_row <- function(parity) {
    oc_summary_by_parity %>% filter(Parity == parity) %>% slice(1)
  }
  
  v <- get_row("Viviparous")
  o <- get_row("Oviparous")
  
  # --- Oxygen effect size ---
  dO2 <- NA_real_
  if (nrow(v) == 1 && nrow(o) == 1) {
    psd <- pooled_sd(v$O2_mean, v$O2_sd, v$n, o$O2_mean, o$O2_sd, o$n)
    if (is.finite(psd) && psd > 0 && is.finite(v$O2_mean) && is.finite(o$O2_mean)) {
      dO2 <- (v$O2_mean - o$O2_mean) / psd
    }
  }
  
  # --- Temperature effect size (may be NA-heavy) ---
  dT <- NA_real_
  if (nrow(v) == 1 && nrow(o) == 1 && v$n_T >= 2 && o$n_T >= 2) {
    psdT <- pooled_sd(v$T_mean, v$T_sd, v$n_T, o$T_mean, o$T_sd, o$n_T)
    if (is.finite(psdT) && psdT > 0 && is.finite(v$T_mean) && is.finite(o$T_mean)) {
      dT <- (v$T_mean - o$T_mean) / psdT
    }
  }
  
  # --- Elevation effect size (also may be sparse) ---
  dE <- NA_real_
  if (nrow(v) == 1 && nrow(o) == 1 && v$n_Elev >= 2 && o$n_Elev >= 2) {
    psdE <- pooled_sd(v$Elev_mean, v$Elev_sd, v$n_Elev, o$Elev_mean, o$Elev_sd, o$n_Elev)
    if (is.finite(psdE) && psdE > 0 && is.finite(v$Elev_mean) && is.finite(o$Elev_mean)) {
      dE <- (v$Elev_mean - o$Elev_mean) / psdE
    }
  }
  
  # Convert effect sizes into *positive magnitudes* for your axis coefficients.
  # Clamp so a weird sparse summary can't explode your parameters.
  mag_O2 <- if (is.finite(dO2)) clamp(abs(dO2), 0.2, 3.0) else 1.0
  mag_T  <- if (is.finite(dT))  clamp(abs(dT),  0.2, 3.0) else 1.0
  mag_E  <- if (is.finite(dE))  clamp(abs(dE),  0.2, 3.0) else 1.0
  
  # Build an overall scaling so the numeric payoffs land in a similar range as before
  # but now anchored to empirical variability.
  # Use oxygen range as a stable anchor (almost always present).
  o2_rng <- if (nrow(v)==1 && nrow(o)==1) {
    rngs <- c(v$O2_max - v$O2_min, o$O2_max - o$O2_min)
    mean(rngs[is.finite(rngs)], na.rm = TRUE)
  } else NA_real_
  if (!is.finite(o2_rng) || o2_rng <= 0) o2_rng <- 5
  
  # Baseline magnitude: a simple monotone function of empirical range
  # (bigger empirical spread -> stronger selection pressure scale)
  base_scale <- clamp(o2_rng / 5, 0.6, 2.0)
  
  # Your prior story: oxygen & cold are strong axes; water is weaker because it's not in the table.
  # We keep the *structure* but let data set the strength.
  # Temperature “cold penalty” axis uses mag_T; oxygen axis uses mag_O2.
  # Elevation (if present) is folded into oxygen/cold strength by a mild multiplier.
  elev_mult <- 1 + 0.15 * (mag_E - 1)  # 1.0 .. ~1.3
  
  # Construct the parameter set
  par <- list(
    # Baselines equal
    V0  = 3.8,
    O0  = 3.8,
    OC0 = 3.8,
    
    lvl_step = 1.0,
    stress_scale = 1.0,
    
    # Viviparity: advantage in cold + low O2 + low water
    V_T_cold = base_scale * elev_mult * (1.0 + 0.6 * mag_T),
    V_T_hot  = base_scale * (0.6 + 0.25 * mag_T),
    
    V_O2_low = base_scale * elev_mult * (1.0 + 0.7 * mag_O2),
    V_O2_hi  = base_scale * (0.25 + 0.10 * mag_O2),
    
    # Water not in table -> keep modest, but still scaled to base_scale
    V_H2O_low= base_scale * 0.7,
    V_H2O_hi = base_scale * 0.2,
    
    # Oviparity: advantage in warmth + high O2 + high water; penalties in cold/hypoxia
    O_T_hot  = base_scale * (1.0 + 0.5 * mag_T),
    O_T_cold = base_scale * elev_mult * (1.1 + 0.8 * mag_T),
    
    O_O2_hi  = base_scale * (0.7 + 0.25 * mag_O2),
    O_O2_low = base_scale * elev_mult * (1.0 + 0.7 * mag_O2),
    
    O_H2O_hi = base_scale * 1.0,
    O_H2O_low= base_scale * 1.2,
    
    bump_M_T   = 0.2 * base_scale,
    bump_M_O2  = 0.1 * base_scale,
    bump_M_H2O = 0.1 * base_scale,
    
    OC_bonus_lowstress = 0.8 * base_scale,
    OC_pen_midstress   = 0.6 * base_scale,
    OC_pen_histress    = 2.0 * base_scale,
    
    # Keep these around for transparency/debug
    .derived = list(
      dO2 = dO2, dT = dT, dE = dE,
      mag_O2 = mag_O2, mag_T = mag_T, mag_E = mag_E,
      base_scale = base_scale, elev_mult = elev_mult, o2_rng = o2_rng
    )
  )
  
  par
}

FIT_PAR <- derive_fit_par(oc_summary_by_parity)

cat("\n=== Derived FIT_PAR (from oc_table) ===\n")
print(FIT_PAR)
#$V0 = 3.8
#$O0 = 3.8
#$OC0 = 3.8
#$lvl_step = 1
#$stress_scale = 1
#$V_T_cold = 2.747193
#$V_T_hot = 1.474755
#$V_O2_low = 2.355257
#$V_O2_hi = 0.54
#$V_H2O_low = 1.4
#$V_H2O_hi = 0.4
#$O_T_hot = 2.549511
#$O_T_cold = 3.180853
#$O_O2_hi = 1.5
#$O_O2_low = 2.355257
#$O_H2O_hi = 2
#$O_H2O_low = 2.4
#$bump_M_T = 0.4
#$bump_M_O2 = 0.2
#$bump_M_H2O = 0.2
#$OC_bonus_lowstress = 1.6
#$OC_pen_midstress = 1.2
#$OC_pen_histress = 4
#$.derived
#$.derived$dO2 = 0.1169904
#$.derived$dT = -0.5495105
#$.derived$dE = 1.220048
#$.derived$mag_O2 = 0.2
#$.derived$mag_T = 0.5495105
#$.derived$mag_E = 1.220048
#$.derived$base_scale = 2
#$.derived$elev_mult = 1.033007
#$.derived$o2_rng = 10.85
cat("\nDerived diagnostics:\n")
print(FIT_PAR$.derived)
#$dO2 = 0.1169904
#$dT = -0.5495105
#$dE = 1.220048
#$mag_O2 = 0.2
#$mag_T = 0.5495105
#$mag_E = 1.220048
#$base_scale = 2
#$elev_mult = 1.033007
#$o2_rng = 10.85

# ---------------------------
# 4) GAME FUNCTIONS (base_fitness, payoff, dynamics, ESS)
# ---------------------------

base_fitness <- function(strategy, O2, H2O, Tmin_lvl, Tmax_lvl, par = FIT_PAR) {
  
  # O2 and H2O still use your 5-level mapping
  O2v  <- lvl_num(O2)        # -2..+2
  H2Ov <- lvl_num(H2O)
  
  # Tmin/Tmax also treated as 5-level stress axes
  Tminv <- lvl_num(Tmin_lvl) # -2..+2 (EL coldest)
  Tmaxv <- lvl_num(Tmax_lvl) # -2..+2 (EH hottest)
  
  step <- par$lvl_step
  if (!is.finite(step) || step <= 0) step <- 1.0
  
  # Cold intensity: colder Tmin => more cold stress
  # EL (-2) => cold=2; M(0)=>0; EH(+2)=>0
  cold <- max(0, -Tminv) / step
  
  # Heat intensity: hotter Tmax => more heat stress
  # EH(+2) => hot=2; M(0)=>0; EL(-2)=>0
  hot  <- max(0,  Tmaxv) / step
  
  o2low<- max(0, -O2v)/ step
  o2hi <- max(0,  O2v)/ step
  wlow <- max(0, -H2Ov)/step
  whi  <- max(0,  H2Ov)/step
  
  if (strategy == "Viviparity") {
    fit <- par$V0
    fit <- fit + par$V_T_cold * cold - par$V_T_hot * hot
    if (Tminv == 0 && Tmaxv == 0) fit <- fit + par$bump_M_T
    
    fit <- fit + par$V_O2_low * o2low - par$V_O2_hi * o2hi
    if (O2v == 0) fit <- fit + par$bump_M_O2
    
    fit <- fit + par$V_H2O_low * wlow - par$V_H2O_hi * whi
    if (H2Ov == 0) fit <- fit + par$bump_M_H2O
    
    return(fit)
  }
  
  if (strategy == "Oviparity") {
    fit <- par$O0
    fit <- fit + par$O_T_hot * hot - par$O_T_cold * cold
    if (Tminv == 0 && Tmaxv == 0) fit <- fit + par$bump_M_T
    
    fit <- fit + par$O_O2_hi * o2hi - par$O_O2_low * o2low
    if (O2v == 0) fit <- fit + par$bump_M_O2
    
    fit <- fit + par$O_H2O_hi * whi - par$O_H2O_low * wlow
    if (H2Ov == 0) fit <- fit + par$bump_M_H2O
    
    return(fit)
  }
  
  if (strategy == "O.C") {
    fit <- par$OC0
    
    # Use combined extremity across (O2, H2O, Tmin, Tmax)
    stress <- abs(O2v) + abs(H2Ov) + abs(Tminv) + abs(Tmaxv)  # now ∈ [0,8]
    
    if (stress <= 2) {
      fit <- fit + par$OC_bonus_lowstress
    } else if (stress <= 5) {
      fit <- fit - par$OC_pen_midstress
    } else {
      fit <- fit - par$OC_pen_histress
    }
    return(fit)
  }
  
  stop("Unknown strategy: ", strategy)
}

# ==============================================================================
# PAYOFF (frequency-dependent fitness) 5^4 = 625
# ==============================================================================
payoff <- function(strategy, freq_vec, O2, H2O, Tmin_lvl, Tmax_lvl, par = FIT_PAR) {
  
  base <- base_fitness(strategy, O2, H2O, Tmin_lvl, Tmax_lvl, par = par)
  
  # Stress used for the O.C frequency-dependent adjustments
  O2v   <- lvl_num(O2)
  H2Ov  <- lvl_num(H2O)
  Tminv <- lvl_num(Tmin_lvl)
  Tmaxv <- lvl_num(Tmax_lvl)
  stress <- abs(O2v) + abs(H2Ov) + abs(Tminv) + abs(Tmaxv)
  
  # Frequency dependent O.C
  if (strategy == "O.C") {
    fp <- freq_vec[STRATS == "O.C"]
    
    if (stress <= 3) {
      if (fp < 0.15) {
        bonus <- -0.3
      } else if (fp < 0.4) {
        bonus <- -0.3 + (fp - 0.15) * 4.0
      } else {
        bonus <- 0.7 + (fp - 0.4) * 1.5
      }
      base <- base + bonus
      
    } else if (stress <= 6) {
      if (fp < 0.3) {
        bonus <- -1.2
      } else if (fp < 0.5) {
        bonus <- -1.2 + (fp - 0.3) * 6.0
      } else {
        bonus <- 0.0 + (fp - 0.5) * 1.5
      }
      base <- base + bonus
      
    } else {
      bonus <- -2.0
      if (fp > 0.5) bonus <- bonus - 1.0
      base <- base + bonus
    }
  }
  
  # If O.C dominates hard, it can suppress oviparity a bit (your old rule)
  if (strategy == "Oviparity") {
    fp <- freq_vec[STRATS == "O.C"]
    if (fp > 0.7) base <- base - 0.5 * (fp - 0.7)
  }
  
  base
}

evolve_to_equilibrium <- function(O2, H2O, Tmin, Tmax,
                                  initial_freq = c(1/3, 1/3, 1/3),
                                  max_gen = 5000,
                                  tolerance = 1e-6,
                                  step_size = 0.1,
                                  par = FIT_PAR) {
  
  initial_freq <- as.numeric(initial_freq)
  if (length(initial_freq) != 3) stop("initial_freq must be length 3 (V, O, O.C).")
  if (any(initial_freq < 0)) stop("initial_freq must be non-negative.")
  initial_freq <- initial_freq / sum(initial_freq)
  
  x <- initial_freq
  trajectory <- matrix(NA, nrow = max_gen + 1, ncol = 3)
  colnames(trajectory) <- STRATS
  trajectory[1,] <- x
  
  for (g in seq_len(max_gen)) {
    fits <- sapply(STRATS, function(s) payoff(s, x, O2, H2O, Tmin, Tmax, par = par))
    mean_fit <- sum(x * fits)
    
    x_new <- x + step_size * x * (fits - mean_fit)
    x_new <- pmax(x_new, 0)
    x_new <- x_new / sum(x_new)
    
    trajectory[g+1,] <- x_new
    
    if (max(abs(x_new - x)) < tolerance) break
    x <- x_new
  }
  
  list(
    equilibrium = as.numeric(x_new),
    trajectory = trajectory[1:(g+1), , drop = FALSE]
  )
}

# ==============================================================================
# ESS CHECK (rare invader test)
# ==============================================================================

test_ESS <- function(freq_vec, O2, H2O, Tmin, Tmax, epsilon = 1e-3, par = FIT_PAR) {
  eq <- as.numeric(freq_vec); names(eq) <- STRATS
  eq <- eq / sum(eq)
  
  # resident mean fitness at eq
  res_fits <- sapply(STRATS, \(s) payoff(s, eq, O2, H2O, Tmin, Tmax, par = par))
  res_mean <- sum(eq * res_fits)
  
  for (m in STRATS) {
    # mutant introduced at epsilon, resident scaled to (1-eps)
    mut <- eq * (1 - epsilon)
    mut[m] <- mut[m] + epsilon
    mut <- mut / sum(mut)
    
    mut_fit <- payoff(m, mut, O2, H2O, Tmin, Tmax, par = par)
    
    if (mut_fit > res_mean + 1e-6) return(FALSE)
  }
  TRUE
}

# ==============================================================================
# MULTI-START ANALYSIS (bistability-ready) + ROBUST BASINS
# ==============================================================================
test_multiple_starts <- function(O2, H2O, Tmin, Tmax, Tavg = "M",
                                 n_random = 20,
                                 uniq_tol = 0.03,
                                 max_gen = 5000,
                                 tolerance = 1e-6,
                                 step_size = 0.1,
                                 par = FIT_PAR,
                                 seed = NULL,
                                 min_basin_size = 3) {
  
  starts <- list(
    # use your data-anchored starting point:
    equal      = initial_from_Tavg(Tavg),
    vivi_bias  = c(0.7,0.15,0.15),
    ovi_bias   = c(0.15,0.7,0.15),
    oc_bias    = c(0.15,0.15,0.7),
    no_care    = c(0.5,0.5,0.0),
    care_only  = c(0.0,0.0,1.0)
  )
  
  if (!is.null(seed)) set.seed(seed)
  for (k in seq_len(n_random)) {
    z <- rexp(3, 1)
    starts[[paste0("rand_", k)]] <- z / sum(z)
  }
  
  eq_tbl <- lapply(names(starts), function(nm) {
    eq <- evolve_to_equilibrium(O2, H2O, Tmin, Tmax,
                                initial_freq = starts[[nm]],
                                max_gen = max_gen,
                                tolerance = tolerance,
                                step_size = step_size,
                                par = par)$equilibrium
    data.frame(Start = nm, Viviparity = eq[1], Oviparity = eq[2], `O.C` = eq[3])
  }) %>% bind_rows()
  
  # ... your clustering code stays the same ...
  
  uniq <- list()
  labels <- character(nrow(eq_tbl))
  
  for (i in seq_len(nrow(eq_tbl))) {
    v <- as.numeric(eq_tbl[i, c("Viviparity","Oviparity","O.C")])
    if (!length(uniq)) {
      uniq[[1]] <- v
      labels[i] <- "E1"
      next
    }
    found <- FALSE
    for (j in seq_along(uniq)) {
      if (max(abs(v - uniq[[j]])) < uniq_tol) {
        labels[i] <- paste0("E", j)
        found <- TRUE
        break
      }
    }
    if (!found) {
      uniq[[length(uniq) + 1]] <- v
      labels[i] <- paste0("E", length(uniq))
    }
  }
  
  eq_tbl$Basin <- labels
  
  uniq_df <- lapply(seq_along(uniq), function(j) {
    v <- uniq[[j]]
    fits <- sapply(STRATS, function(s) payoff(s, v, O2, H2O, Tmin, Tmax, par = par))
    data.frame(
      Basin = paste0("E", j),
      Viviparity = v[1], Oviparity = v[2], `O.C` = v[3],
      MeanFitness = sum(v * fits)
    )
  }) %>% bind_rows()
  
  best_row <- uniq_df %>% slice_max(MeanFitness, n = 1)
  
  basin_counts <- as.data.frame(table(eq_tbl$Basin), stringsAsFactors = FALSE)
  colnames(basin_counts) <- c("Basin","N_Starts")
  uniq_df <- uniq_df %>% left_join(basin_counts, by = "Basin")
  uniq_df$N_Starts[is.na(uniq_df$N_Starts)] <- 0
  
  n_all <- nrow(uniq_df)
  n_robust <- sum(uniq_df$N_Starts >= min_basin_size)
  
  list(
    best_equilibrium = as.numeric(best_row[1, c("Viviparity","Oviparity","O.C")]),
    n_equilibria_all = n_all,
    n_equilibria_robust = n_robust,
    all_equilibria = uniq,
    basin_map = eq_tbl,
    uniq_table = uniq_df
  )
}


# ---------------------------
# 5) RUN SIMS ACROSS OBSERVED ENVIRONMENTS (env5 id, but uses env4 axes)
# ---------------------------
need_recompute <- TRUE
if (!need_recompute && file.exists(CACHE_FILE)) {
  results_df <- readRDS(CACHE_FILE)
  need_recompute <- !all(c("O2","H2O","Tmin","Tmax","Tavg","env4","env5") %in% names(results_df))
}

if (need_recompute) {
  message("Running evolutionary simulations across observed environments...")
  results_list <- vector("list", nrow(env_grid))
  
  for (i in seq_len(nrow(env_grid))) {
    env <- env_grid[i,]
    
    res <- test_multiple_starts(
      env$O2, env$H2O, env$Tmin, env$Tmax,
      Tavg = env$Tavg,
      n_random = 20, uniq_tol = 0.03,
      par = FIT_PAR, seed = 1, min_basin_size = 3
    )
    
    eq <- res$best_equilibrium
    
    results_list[[i]] <- data.frame(
      env4 = env$env4,
      env5 = env$env5,
      O2 = env$O2, H2O = env$H2O, Tmin = env$Tmin, Tmax = env$Tmax, Tavg = env$Tavg,
      Viviparity = eq[1], Oviparity = eq[2], `O.C` = eq[3],
      Dominant = STRATS[which.max(eq)],
      N_Basins_All    = res$n_equilibria_all,
      N_Basins_Robust = res$n_equilibria_robust,
      Is_Bistable_All    = res$n_equilibria_all >= 2,
      Is_Bistable_Robust = res$n_equilibria_robust >= 2,
      Environmental_Stress = env_stress(env$O2, env$H2O, env$Tmin, env$Tmax),
      Is_ESS = test_ESS(eq, env$O2, env$H2O, env$Tmin, env$Tmax, par = FIT_PAR),
      stringsAsFactors = FALSE
    )
  }
  
  results_df <- bind_rows(results_list)
  saveRDS(results_df, CACHE_FILE)
}

write.csv(results_df, file.path(OUT_DIR, "results_df_all_observed_envs.csv"), row.names = FALSE)

# ---------------------------
# 6) LANDSCAPES FOR MANY ENVS (env4 only)
# ---------------------------
mk_landscape <- function(env4, par = FIT_PAR) {
  parts <- strsplit(env4, "-")[[1]]
  stopifnot(length(parts) == 4)
  
  O2 <- parts[1]; H2O <- parts[2]; Tmin <- parts[3]; Tmax <- parts[4]
  freq_range <- seq(0, 1, by = 0.02)
  
  out <- lapply(freq_range, function(fp) {
    rem <- 1 - fp
    freq_vec <- c(rem*0.5, rem*0.5, fp)
    
    data.frame(
      env4 = env4,
      Freq_OC = fp,
      Viviparity = payoff("Viviparity", freq_vec, O2, H2O, Tmin, Tmax, par = par),
      Oviparity  = payoff("Oviparity",  freq_vec, O2, H2O, Tmin, Tmax, par = par),
      `O.C`      = payoff("O.C",        freq_vec, O2, H2O, Tmin, Tmax, par = par)
    )
  })
  
  bind_rows(out) %>%
    pivot_longer(cols = c(Viviparity, Oviparity, `O.C`),
                 names_to = "Strategy", values_to = "Fitness"
    )
}

env4_observed <- sort(unique(results_df$env4))
env4_observed #82
landscape_df <- bind_rows(lapply(env4_observed, mk_landscape, par = FIT_PAR))
landscape_df #12,383 more rows
# A tibble: 12,393 × 4
#env4        Freq_OC Strategy   Fitness
#<chr>         <dbl> <chr>        <dbl>
#  1 EH-EH-EH-EH    0    Viviparity   -1.03
#2 EH-EH-EH-EH    0    Oviparity    15.9 
#3 EH-EH-EH-EH    0    O.C          -2.2 
#4 EH-EH-EH-EH    0.02 Viviparity   -1.03
#5 EH-EH-EH-EH    0.02 Oviparity    15.9 
#6 EH-EH-EH-EH    0.02 O.C          -2.2 
#7 EH-EH-EH-EH    0.04 Viviparity   -1.03
#8 EH-EH-EH-EH    0.04 Oviparity    15.9 
#9 EH-EH-EH-EH    0.04 O.C          -2.2 
#10 EH-EH-EH-EH    0.06 Viviparity   -1.03

# Example “how to read this” (one baseline env)
baseline_env4 <- "M-M-M-M"
baseline_plot_df <- mk_landscape(baseline_env4, par = FIT_PAR)

p_base <- ggplot(baseline_plot_df, aes(Freq_OC, Fitness, color = Strategy)) +
  geom_line(linewidth = 1.1) +
  scale_color_manual(values = STRAT_COLORS) +
  theme_minimal(base_size = 12) +
  labs(title = paste("How to read: Fitness vs O.C frequency —", baseline_env4))
print(p_base)

# Facet all observed envs (will be huge; usually save to file or subset first)
p_all <- ggplot(landscape_df, aes(Freq_OC, Fitness, color = Strategy)) +
  geom_line(linewidth = 0.7) +
  facet_wrap(~ env4, scales = "free_y") +
  scale_color_manual(values = STRAT_COLORS) +
  theme_minimal(base_size = 10)
 print(p_all) # usually too many panels to view interactively

# ---------------------------
# 7) PICK “BEST” ENVS FOR FIGURES (defensible rules)
# ---------------------------
pick_df <- results_df %>%
  group_by(Dominant) %>%
  slice_max(order_by = pmax(Viviparity, Oviparity, `O.C`), n = 1) %>%
  ungroup()

stress_low  <- results_df %>% slice_min(Environmental_Stress, n = 1)
stress_high <- results_df %>% slice_max(Environmental_Stress, n = 1)
oc_low      <- results_df %>% slice_min(`O.C`, n = 1)
oc_high     <- results_df %>% slice_max(`O.C`, n = 1)
bistable    <- results_df %>% filter(Is_Bistable_Robust) %>% slice_head(n = 1)

chosen_env4 <- unique(c(
  baseline_env4,
  pick_df$env4,
  stress_low$env4, stress_high$env4,
  oc_low$env4, oc_high$env4,
  bistable$env4
))

chosen_landscapes <- bind_rows(lapply(chosen_env4, mk_landscape, par = FIT_PAR))

p_chosen <- ggplot(chosen_landscapes, aes(Freq_OC, Fitness, color = Strategy)) +
  geom_line(linewidth = 1.1) +
  facet_wrap(~ env4, scales = "free_y") +
  scale_color_manual(values = STRAT_COLORS) +
  theme_minimal(base_size = 12) +
  labs(title = "Selected environments (rule-based) — Fitness landscapes")
print(p_chosen)

#-------------------------------------------------------------------------------
# 8) MAKING ANOVAS
#-------------------------------------------------------------------------------

results_df2 <- results_df %>%
  mutate(
    O2   = factor(O2,   levels = lev5),
    H2O  = factor(H2O,  levels = lev5),
    Tmin = factor(Tmin, levels = lev5),
    Tmax = factor(Tmax, levels = lev5),
    Tavg = factor(Tavg, levels = lev5)
  )

fit_V <- aov(Viviparity ~ O2 + H2O + Tmin + Tmax, data = results_df2)
summary(fit_V)

fit_O <- aov(Oviparity ~ O2 + H2O + Tmin + Tmax, data = results_df2)
summary(fit_O)

fit_OC <- aov(`O.C` ~ O2 + H2O + Tmin + Tmax, data = results_df)
summary(fit_OC)

install.packages("effectsize")
library(effectsize)

eta_squared(fit_V)
eta_squared(fit_O)
eta_squared(fit_OC)

#-------------------------------------------------------------------------------
# theoretical vs observed
#-------------------------------------------------------------------------------
lev5 <- c("EL","L","M","H","EH")

env4_theoretical <- expand.grid(
  O2 = lev5,
  H2O = lev5,
  Tmin = lev5,
  Tmax = lev5
)

n_theoretical <- nrow(env4_theoretical)

env4_observed <- results_df2 %>%
  distinct(O2, H2O, Tmin, Tmax)

n_observed <- nrow(env4_observed)

cat("Theoretical env4 combos:", n_theoretical, "\n")
cat("Observed env4 combos:", n_observed, "\n")

#-------------------------------------------------------------------------------
# save all plots
#-------------------------------------------------------------------------------
save_plot <- function(plot_obj, filename, width = 8, height = 6) {
  
  # Ensure output directory exists
  if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)
  
  # PDF
  ggsave(
    filename = file.path(OUT_DIR, paste0(filename, ".pdf")),
    plot = plot_obj,
    width = width,
    height = height
  )
  
  # JPEG
  ggsave(
    filename = file.path(OUT_DIR, paste0(filename, ".jpg")),
    plot = plot_obj,
    width = width,
    height = height,
    dpi = 300
  )
}

for (env in unique(landscape_df$env4)) {
  df_env <- landscape_df %>% filter(env4 == env)
  
  p_env <- ggplot(df_env, aes(Freq_OC, Fitness, color = Strategy)) +
    geom_line(linewidth = 1.1) +
    scale_color_manual(values = STRAT_COLORS) +
    theme_minimal(base_size = 12) +
    labs(title = paste("Fitness Landscape:", env))
  save_plot(p_env, paste0("landscape_", env), width = 7, height = 5)
}

#-------------------------------------------------------------------------------
# regression
#-------------------------------------------------------------------------------

lm_V <- lm(Viviparity ~ Environmental_Stress, data = results_df2)
summary(lm_V)
#Residuals:
#Min      1Q  Median      3Q     Max 
#-0.7984 -0.4943  0.2396  0.4296  0.6577 

#Coefficients:
#  Estimate Std. Error t value Pr(>|t|)  
#(Intercept)           0.19033    0.14529   1.310   0.1936  
#Environmental_Stress  0.07601    0.03111   2.443   0.0166 *
#Residual standard error: 0.4888 on 88 degrees of freedom
#Multiple R-squared:  0.06352,	Adjusted R-squared:  0.05288 
#F-statistic: 5.969 on 1 and 88 DF,  p-value: 0.01656

lm_OC <- lm(`O.C` ~ Environmental_Stress, data = results_df2)
summary(lm_OC)
#Residuals:
#  Min       1Q   Median       3Q      Max 
#-0.31375 -0.13312 -0.04280  0.04752  0.68624 

#Coefficients:
#  Estimate Std. Error t value Pr(>|t|)    
#(Intercept)           0.49439    0.07816   6.325 1.02e-08 ***
#  Environmental_Stress -0.09032    0.01674  -5.397 5.67e-07 ***
#Residual standard error: 0.263 on 88 degrees of freedom
#Multiple R-squared:  0.2487,	Adjusted R-squared:  0.2401 
#F-statistic: 29.12 on 1 and 88 DF,  p-value: 5.675e-07

lm_O <- lm(Oviparity ~ Environmental_Stress, data = results_df)
summary(lm_O)
#Residuals:
#Min      1Q  Median      3Q     Max 
#-0.4298 -0.3868 -0.3510  0.6132  0.6561 
# Estimate Std. Error t value Pr(>|t|)  
#(Intercept)           0.31528    0.14555   2.166    0.033 *
#  Environmental_Stress  0.01431    0.03117   0.459    0.647  

#===============================================================================
# ANOVA interactions
#===============================================================================
lev5 <- c("EL","L","M","H","EH")

results_df3 <- results_df %>%
  mutate(
    O2   = factor(O2,   levels = lev5),
    H2O  = factor(H2O,  levels = lev5),
    Tmin = factor(Tmin, levels = lev5),
    Tmax = factor(Tmax, levels = lev5)
  )
head(results_df3)
results_df3
# env4          env5 O2 H2O Tmin Tmax Tavg   Viviparity    Oviparity           O.C
#1 L-EH-EH-EH L-EH-EH-EH-EH  L  EH   EH   EH   EH 1.634436e-07 9.999998e-01  0.000000e+00
#2  EL-EH-H-H   EL-EH-H-H-H EL  EH    H    H    H 9.999843e-01 1.568640e-05 1.524334e-120
#3   EL-H-H-H    EL-H-H-H-H EL   H    H    H    H 9.999983e-01 1.706623e-06  1.877905e-13
#4 H-EH-EH-EH H-EH-EH-EH-EH  H  EH   EH   EH   EH 0.000000e+00 1.000000e+00  0.000000e+00
#5 M-EH-EH-EH M-EH-EH-EH-EH  M  EH   EH   EH   EH 0.000000e+00 1.000000e+00  0.000000e+00
#6   EL-H-M-H    EL-H-M-H-M EL   H    M    H    M 9.999980e-01 2.021674e-06  1.697289e-12
#Dominant N_Basins_All N_Basins_Robust Is_Bistable_All Is_Bistable_Robust
#1  Oviparity            2               1            TRUE              FALSE
#2 Viviparity            2               1            TRUE              FALSE
#3 Viviparity            2               1            TRUE              FALSE
#4  Oviparity            2               1            TRUE              FALSE
#5  Oviparity            2               1            TRUE              FALSE
#6 Viviparity            2               1            TRUE              FALSE
#Environmental_Stress Is_ESS
#1                    7  FALSE
#2                    6  FALSE
#3                    5  FALSE
#4                    7   TRUE
#5                    6   TRUE
#6                    4  FALSE

form_2way <- as.formula("Y ~ (O2 + H2O + Tmin + Tmax)^2")

mV <- lm(update(form_2way, Viviparity ~ .), data = results_df3)
#has NAs
mO <- lm(update(form_2way, Oviparity ~ .), data = results_df3)
#Has NAs
mOC <- lm(update(form_2way, `O.C` ~ .), data = results_df3)
#Has NAs

# Dominant should be a factor (3 classes)
results_df4 <- results_df3 %>%
  mutate(
    Dominant = factor(Dominant, levels = c("Viviparity","Oviparity","O.C")),
    O2   = factor(O2,   levels = lev5),
    H2O  = factor(H2O,  levels = lev5),
    Tmin = factor(Tmin, levels = lev5),
    Tmax = factor(Tmax, levels = lev5)
  )

table(results_df4$Dominant)

options(contrasts = c("contr.sum", "contr.poly"))

m_dom <- nnet::multinom(Dominant ~ (O2 + H2O + Tmin + Tmax)^2,
                        data = results_df3,
                        trace = FALSE)

# Likelihood-ratio tests for terms (Type II is usually more stable than Type III)
anova_dom <- car::Anova(m_dom, type = 2, test.statistic = "LR")
anova_dom
#Response: Dominant
#LR Chisq Df Pr(>Chisq)    
#O2          27.726  8  0.0005293 ***
#  H2O         37.653  8  8.724e-06 ***
#  Tmin        22.547  8  0.0039965 ** 
#  Tmax        10.411  8  0.2373802    
#O2:H2O       0.000 32  1.0000000    
#O2:Tmin      0.000 32  1.0000000    
#O2:Tmax      0.000 32  1.0000000    
#H2O:Tmin     0.000 32  1.0000000    
#H2O:Tmax     0.000 32  1.0000000    
#Tmin:Tmax    0.000 32  1.0000000 

extract_dom_interactions <- function(anova_obj) {
  tab <- as.data.frame(anova_obj)
  tab$Term <- rownames(tab)
  rownames(tab) <- NULL
  
  # car::Anova(multinom, test="LR") columns look like these:
  stopifnot(all(c("Df") %in% names(tab)))
  lr_col <- grep("Chisq", names(tab), value = TRUE)[1]      # e.g., "LR Chisq"
  p_col  <- grep("^Pr\\(", names(tab), value = TRUE)[1]     # e.g., "Pr(>Chisq)"
  
  tab %>%
    dplyr::filter(grepl(":", Term)) %>%
    dplyr::transmute(
      Outcome = "Dominant (multinom)",
      Interaction = Term,
      Df = Df,
      LR = .data[[lr_col]],
      p  = .data[[p_col]]
    ) %>%
    dplyr::mutate(
      # guard against tiny negative LR from numerical noise
      LR = pmax(LR, 0),
      p_BH = p.adjust(p, method = "BH")
    ) %>%
    dplyr::arrange(p)
}

dom_int_table <- extract_dom_interactions(anova_dom)
dom_int_table <- as.data.frame(dom_int_table)
dom_int_table
# Outcome Interaction Df            LR p p_BH
#1 Dominant (multinom)   Tmin:Tmax 32  1.744918e-05 1    1
#2 Dominant (multinom)      O2:H2O 32  4.360750e-06 1    1
#3 Dominant (multinom)     O2:Tmin 32 -2.559324e-05 1    1
#4 Dominant (multinom)     O2:Tmax 32  1.830337e-05 1    1
#5 Dominant (multinom)    H2O:Tmin 32 -8.545489e-06 1    1
#6 Dominant (multinom)    H2O:Tmax 32 -7.900982e-06 1    1

all_interactions <- dom_int_table

all_interactions
#Outcome Interaction Df           LR p p_BH
#1 Dominant (multinom)      O2:H2O 32 0.000000e+00 1    1
#2 Dominant (multinom)     O2:Tmin 32 1.185531e-05 1    1
#3 Dominant (multinom)     O2:Tmax 32 3.772654e-05 1    1
#4 Dominant (multinom)    H2O:Tmin 32 8.954426e-06 1    1
#5 Dominant (multinom)    H2O:Tmax 32 0.000000e+00 1    1
#6 Dominant (multinom)   Tmin:Tmax 32 2.840116e-05 1    1

names(as.data.frame(anova_dom))
m_dom <- nnet::multinom(Dominant ~ (O2 + H2O + Tmin + Tmax)^2,
                        data = results_df4,
                        trace = FALSE)
anova_dom <- car::Anova(m_dom, type = 2, test.statistic = "LR")
anova_dom
#          LR Chisq Df Pr(>Chisq)    
#O2          27.726  8  0.0005293 ***
#H2O         37.653  8  8.724e-06 ***
#Tmin        22.547  8  0.0039965 ** 

summary_table <- results_df3 %>%
  summarise(
    Total = n(),
    Viviparity = mean(Dominant == "Viviparity") * 100,
    Oviparity = mean(Dominant == "Oviparity") * 100,
    OC = mean(Dominant == "O.C") * 100,
    Bistable = sum(Is_Bistable_All),
    Robust_Bistable = sum(Is_Bistable_Robust),
    ESS = sum(Is_ESS),
    Non_ESS = sum(!Is_ESS),
    Stress_Min = min(Environmental_Stress),
    Stress_Max = max(Environmental_Stress)
  )
summary_table
#===============================================================================
#DOMINANT STRATEGY HEATMAP
#===============================================================================
results_df_pub <- results_df %>%
  mutate(
    O2   = factor(O2,   levels = lev5),
    H2O  = factor(H2O,  levels = lev5),
    Tmin = factor(Tmin, levels = lev5),
    Tmax = factor(Tmax, levels = lev5),
    Dominant = factor(Dominant, levels = c("Viviparity", "Oviparity", "O.C"))
  )

p_dom_heat <- ggplot(results_df_pub, aes(x = O2, y = H2O, fill = Dominant)) +
  geom_tile(color = "white") +
  facet_grid(Tmin ~ Tmax) +
  scale_fill_manual(values = STRAT_COLORS) +
  theme_minimal(base_size = 12) +
  labs(
    title = "Dominant reproductive strategy across environments",
    x = "Oxygen level",
    y = "Water level"
  )

print(p_dom_heat)

### now add stars

results_df_pub <- results_df %>%
  mutate(
    O2   = factor(O2,   levels = lev5),
    H2O  = factor(H2O,  levels = lev5),
    Tmin = factor(Tmin, levels = lev5),
    Tmax = factor(Tmax, levels = lev5),
    Dominant = factor(Dominant, levels = c("Viviparity", "Oviparity", "O.C"))
  )

heat_df <- results_df_pub %>%
  mutate(
    Dominant = as.character(Dominant),
    Bistable = ifelse(Is_Bistable_Robust, "Bistable", "Single equilibrium")
  ) %>%
  complete(O2, H2O, Tmin, Tmax) %>%
  mutate(
    Dominant = replace_na(Dominant, "Not observed"),
    Bistable = replace_na(Bistable, "Not observed")
  )

STRAT_COLORS2 <- c(
  "Viviparity"   = "#D55E00",
  "Oviparity"    = "#0072B2",
  "O.C"          = "#009E73",
  "Not observed" = "grey90"
)

p_dom_bistable <- ggplot(heat_df, aes(x = O2, y = H2O, fill = Dominant)) +
  geom_tile(color = "white") +
  geom_point(
    data = subset(heat_df, Bistable == "Bistable"),
    aes(x = O2, y = H2O),
    inherit.aes = FALSE,
    shape = 8, size = 2.8, color = "black"
  ) +
  facet_grid(Tmin ~ Tmax) +
  scale_fill_manual(values = STRAT_COLORS2) +
  scale_y_discrete(limits = rev(lev5)) +
  theme_minimal(base_size = 12) +
  labs(
    title = "Dominant reproductive strategy across environments",
    subtitle = "Black stars indicate robust bistable environments",
    x = "Oxygen level",
    y = "Water level",
    fill = "Outcome"
  )

print(p_dom_bistable)

#===============================================================================
#mean equilibrium by stress level
#===============================================================================

stress_long <- results_df %>%
  select(Environmental_Stress, Viviparity, Oviparity, `O.C`) %>%
  pivot_longer(
    cols = c(Viviparity, Oviparity, `O.C`),
    names_to = "Strategy",
    values_to = "Equilibrium"
  )

stress_summary <- stress_long %>%
  group_by(Environmental_Stress, Strategy) %>%
  summarise(
    mean_eq = mean(Equilibrium, na.rm = TRUE),
    sd_eq   = sd(Equilibrium, na.rm = TRUE),
    n       = n(),
    se_eq   = sd_eq / sqrt(n),
    .groups = "drop"
  )

p_stress_mean <- ggplot(stress_summary,
                        aes(x = Environmental_Stress, y = mean_eq, color = Strategy)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = mean_eq - se_eq, ymax = mean_eq + se_eq),
                width = 0.15, alpha = 0.6) +
  scale_color_manual(values = STRAT_COLORS) +
  scale_x_continuous(breaks = 0:8) +
  theme_minimal(base_size = 12) +
  labs(
    title = "Equilibrium strategy frequencies across environmental stress",
    subtitle = "Points show mean eq.freq across environments at each stress level",
    x = "Environmental stress",
    y = "Mean equilibrium frequency",
    color = "Strategy"
  )

print(p_stress_mean)

#### boxplot by stress KIND OF BAD DON'T USE IT
p_stress_box <- ggplot(stress_long,
                       aes(x = factor(Environmental_Stress), y = Equilibrium, fill = Strategy)) +
  geom_boxplot(outlier.alpha = 0.4) +
  facet_wrap(~ Strategy, ncol = 1) +
  scale_fill_manual(values = STRAT_COLORS) +
  theme_minimal(base_size = 12) +
  labs(
    title = "Distribution of eq.freq across environmental stress",
    x = "Environmental stress",
    y = "Equilibrium frequency"
  ) +
  theme(legend.position = "none")

print(p_stress_box)

# ------------------------------------------------------------------------------
# SCATTERPLOT: Predicted Number of Parity by Elevation
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# FIGURE 1: Model-predicted parity trends across elevation
# ------------------------------------------------------------------------------

# Build plotting dataframe
plot_df <- clim %>%
  transmute(
    Elev,
    O2   = O2_level,
    H2O  = H2O_level,
    Tmin = Tmin_level,
    Tmax = Tmax_level,
    Tavg = Tavg_level
  ) %>%
  left_join(
    results_df %>%
      select(O2, H2O, Tmin, Tmax, Tavg, Viviparity, Oviparity, `O.C`),
    by = c("O2", "H2O", "Tmin", "Tmax", "Tavg")
  ) %>%
  pivot_longer(
    cols = c(Viviparity, Oviparity, `O.C`),
    names_to = "Parity",
    values_to = "Predicted_N"
  )

p_model <- ggplot(plot_df, aes(x = Elev, y = Predicted_N, color = Parity)) +
  geom_smooth(
    method = "loess",
    se = FALSE,
    linetype = "dashed",
    linewidth = 1.2
  ) +
  scale_color_manual(values = c(
    "Oviparity"  = "#0072B2",
    "Viviparity" = "#D55E00",
    "O.C"        = "#009E73"
  )) +
  labs(
    title = "Predicted Number of Parity by Elevation",
    x = "Elevation",
    y = "Predicted Number",
    color = "Parity"
  ) +
  theme_minimal(base_size = 14)

print(p_model)

# ------------------------------------------------------------------------------
# FIGURE 2: Empirical species data across elevation
# ------------------------------------------------------------------------------

empirical_df <- oc_table_raw %>%
  mutate(
    Parity = case_when(
      str_detect(Group, "Oviparous") ~ "Oviparity",
      str_detect(Group, "Viviparous") ~ "Viviparity",
      TRUE ~ NA_character_
    ),
    Elevation_m_text = str_replace_all(Elevation_m_text, "–", "-"),
    Elev_min = as.numeric(str_extract(Elevation_m_text, "^[0-9]+")),
    Elev_max = as.numeric(str_extract(Elevation_m_text, "(?<=-)[0-9]+")),
    Elev = case_when(
      !is.na(Elev_min) & !is.na(Elev_max) ~ (Elev_min + Elev_max) / 2,
      !is.na(Elev_min) ~ Elev_min,
      TRUE ~ NA_real_
    ),
    Temperature_C_num = case_when(
      str_detect(as.character(Temperature_C), "-") ~
        (as.numeric(str_extract(as.character(Temperature_C), "^[0-9.]+")) +
           as.numeric(str_extract(as.character(Temperature_C), "(?<=-)[0-9.]+"))) / 2,
      TRUE ~ suppressWarnings(as.numeric(as.character(Temperature_C)))
    )
  ) %>%
  filter(!is.na(Elev), !is.na(Parity), !is.na(O2_capacity_volpct))

p_empirical <- ggplot(empirical_df, aes(x = Elev, y = O2_capacity_volpct, shape = Parity)) +
  geom_jitter(width = 35, height = 0, size = 3, alpha = 0.85, color = "black") +
  scale_shape_manual(values = c(
    "Oviparity" = 16,
    "Viviparity" = 17
  )) +
  labs(
    title = "Empirical Oxygen Capacity by Elevation",
    x = "Elevation (m)",
    y = "Oxygen Capacity (vol%)",
    shape = "Parity"
  ) +
  theme_minimal(base_size = 14)

print(p_empirical)

# ------------------------------------------------------------------------------
# STACKED BAR PLOT: Dominant strategy shifts across environmental stress
# ------------------------------------------------------------------------------

library(dplyr)
library(ggplot2)

# colorblind-friendly palette
STRAT_COLORS <- c(
  "Viviparity" = "#D55E00",
  "Oviparity"  = "#0072B2",
  "O.C"        = "#009E73"
)

# summarize dominant strategy proportions within each stress level
stress_dom_df <- results_df %>%
  group_by(Environmental_Stress, Dominant) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(Environmental_Stress) %>%
  mutate(Prop = n / sum(n)) %>%
  ungroup() %>%
  mutate(
    Dominant = factor(Dominant, levels = c("O.C", "Oviparity", "Viviparity"))
  )

# stacked proportional bar plot
p_stress_bar <- ggplot(stress_dom_df,
                       aes(x = factor(Environmental_Stress),
                           y = Prop,
                           fill = Dominant)) +
  geom_col(color = "black", width = 0.8) +
  scale_fill_manual(values = STRAT_COLORS) +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title = "Dominant Strategy Shifts Across Environmental Stress",
    x = "Environmental Stress",
    y = "Proportion of Environments",
    fill = "Dominant Strategy"
  ) +
  theme_minimal(base_size = 12)

print(p_stress_bar)


# ============================================================
# SENSITIVITY ANALYSIS: perturb FIT_PAR parameter groups
# ============================================================

run_game_for_par <- function(par_obj, scenario_name = "baseline") {
  results_list <- vector("list", nrow(env_grid))
  
  for (i in seq_len(nrow(env_grid))) {
    env <- env_grid[i,]
    
    res <- test_multiple_starts(
      env$O2, env$H2O, env$Tmin, env$Tmax,
      Tavg = env$Tavg,
      n_random = 20,
      uniq_tol = 0.03,
      par = par_obj,
      seed = 1,
      min_basin_size = 3
    )
    
    eq <- res$best_equilibrium
    
    results_list[[i]] <- data.frame(
      Scenario = scenario_name,
      env4 = env$env4,
      env5 = env$env5,
      O2 = env$O2,
      H2O = env$H2O,
      Tmin = env$Tmin,
      Tmax = env$Tmax,
      Tavg = env$Tavg,
      Viviparity = eq[1],
      Oviparity = eq[2],
      `O.C` = eq[3],
      Dominant = STRATS[which.max(eq)],
      N_Basins_All = res$n_equilibria_all,
      N_Basins_Robust = res$n_equilibria_robust,
      Is_Bistable_All = res$n_equilibria_all >= 2,
      Is_Bistable_Robust = res$n_equilibria_robust >= 2,
      Environmental_Stress = env_stress(env$O2, env$H2O, env$Tmin, env$Tmax),
      Is_ESS = test_ESS(eq, env$O2, env$H2O, env$Tmin, env$Tmax, par = par_obj),
      stringsAsFactors = FALSE
    )
  }
  
  dplyr::bind_rows(results_list)
}


perturb_par_group <- function(base_par, group_name, multiplier) {
  par_new <- base_par
  
  groups <- list(
    Temperature = c("V_T_cold", "V_T_hot", "O_T_hot", "O_T_cold"),
    Oxygen = c("V_O2_low", "V_O2_hi", "O_O2_hi", "O_O2_low"),
    Water = c("V_H2O_low", "V_H2O_hi", "O_H2O_hi", "O_H2O_low"),
    ParentalCare = c("OC_bonus_lowstress", "OC_pen_midstress", "OC_pen_histress")
  )
  
  params_to_change <- groups[[group_name]]
  
  if (is.null(params_to_change)) {
    stop("Unknown group name: ", group_name)
  }
  
  for (p in params_to_change) {
    par_new[[p]] <- par_new[[p]] * multiplier
  }
  
  par_new
}

# Baseline + perturbation scenarios
sensitivity_scenarios <- expand.grid(
  Group = c("Temperature", "Oxygen", "Water", "ParentalCare"),
  Multiplier = c(0.75, 0.90, 1.10, 1.25),
  stringsAsFactors = FALSE
)

# Run baseline once
sens_results <- list()
sens_results[["Baseline_1.00"]] <- run_game_for_par(FIT_PAR, "Baseline_1.00")

# Run sensitivity scenarios
for (i in seq_len(nrow(sensitivity_scenarios))) {
  group_i <- sensitivity_scenarios$Group[i]
  mult_i  <- sensitivity_scenarios$Multiplier[i]
  
  scenario_name <- paste0(group_i, "_x", mult_i)
  message("Running sensitivity scenario: ", scenario_name)
  
  par_i <- perturb_par_group(FIT_PAR, group_i, mult_i)
  sens_results[[scenario_name]] <- run_game_for_par(par_i, scenario_name)
}

# ============================================================
# INDIVIDUAL PARAMETER TESTS
# ============================================================

perturb_single_parameter <- function(base_par, param_name, multiplier) {
  par_new <- base_par
  
  if (is.null(par_new[[param_name]])) {
    stop("Unknown parameter name: ", param_name)
  }
  
  par_new[[param_name]] <- par_new[[param_name]] * multiplier
  par_new
}

single_param_scenarios <- expand.grid(
  Parameter = c(
    "V_T_cold", "V_T_hot",
    "V_O2_low", "V_O2_hi",
    "V_H2O_low", "V_H2O_hi",
    "O_T_hot", "O_T_cold",
    "O_O2_hi", "O_O2_low",
    "O_H2O_hi", "O_H2O_low",
    "OC_bonus_lowstress",
    "OC_pen_midstress",
    "OC_pen_histress"
  ),
  Multiplier = c(0.75, 0.90, 1.10, 1.25),
  stringsAsFactors = FALSE
)

nrow(sensitivity_scenarios)
nrow(single_param_scenarios)

# Quick sanity check
cat("Group scenarios:", nrow(sensitivity_scenarios), "\n")
cat("Single-parameter scenarios:", nrow(single_param_scenarios), "\n")
cat("Total scenarios:", 1 + nrow(sensitivity_scenarios) + nrow(single_param_scenarios), "\n")

# Run individual parameter scenarios
for (i in seq_len(nrow(single_param_scenarios))) {
  param_i <- single_param_scenarios$Parameter[i]
  mult_i  <- single_param_scenarios$Multiplier[i]
  
  scenario_name <- paste0(param_i, "_x", mult_i)
  message("Running individual sensitivity scenario: ", scenario_name)
  
  par_i <- perturb_single_parameter(FIT_PAR, param_i, mult_i)
  sens_results[[scenario_name]] <- run_game_for_par(par_i, scenario_name)
}

sensitivity_df <- dplyr::bind_rows(sens_results)



write.csv(
  sensitivity_df,
  file.path(OUT_DIR, "sensitivity_results_all.csv"),
  row.names = FALSE
)


# ============================================================
# COLLAPSE TO 81 UNIQUE GAME ENVIRONMENTS
# ============================================================

sensitivity_df_81 <- sensitivity_df %>%
  group_by(Scenario, env4) %>%
  slice_max(
    order_by = pmax(Viviparity, Oviparity, `O.C`),
    n = 1,
    with_ties = FALSE
  ) %>%
  ungroup()

# Check that each scenario now has 81 environments
sensitivity_df_81 %>%
  count(Scenario)

write.csv(
  sensitivity_df_81,
  file.path(OUT_DIR, "sensitivity_results_81_env4.csv"),
  row.names = FALSE
)


# ============================================================
# SUMMARIZE SENSITIVITY RESULTS ACROSS 81 UNIQUE ENV4 STATES
# ============================================================

sensitivity_summary_81 <- sensitivity_df_81 %>%
  group_by(Scenario) %>%
  summarise(
    Total_Environments = n(),
    Viviparity_Dominant = sum(Dominant == "Viviparity"),
    Oviparity_Dominant = sum(Dominant == "Oviparity"),
    OC_Dominant = sum(Dominant == "O.C"),
    Viviparity_Percent = mean(Dominant == "Viviparity") * 100,
    Oviparity_Percent = mean(Dominant == "Oviparity") * 100,
    OC_Percent = mean(Dominant == "O.C") * 100,
    Mean_Viviparity = mean(Viviparity),
    Mean_Oviparity = mean(Oviparity),
    Mean_OC = mean(`O.C`),
    ESS_Count = sum(Is_ESS),
    Non_ESS_Count = sum(!Is_ESS),
    Robust_Bistable_Count = sum(Is_Bistable_Robust),
    .groups = "drop"
  )

write.csv(
  sensitivity_summary_81,
  file.path(OUT_DIR, "sensitivity_summary_81_env4.csv"),
  row.names = FALSE
)

print(sensitivity_summary_81)

write.csv(
  sensitivity_summary_81,
  file.path(OUT_DIR, "sensitivity_summary_81.csv"),
  row.names = FALSE
)

View(sensitivity_summary_81)





















