# ============================================================
# MMLU initialization reproduced in R, line by line
# ============================================================
# This is a standalone inspection script.  It does NOT source or modify
# the Python estimator.  The code is intentionally written sequentially
# rather than wrapped into helper functions, so objects can be inspected
# after each block in an interactive R session.
#
# It reproduces the current initialization logic:
#   Proposed:
#     subject accuracy -> theta_t^(0)
#     theta_t^(0) -> bounded itemwise logistic a_t^(0), d_t^(0)
#     {theta_t^(0)} -> unweighted sine-center theta_G^(0)
#
#   Global:
#     same subjectwise theta_t^(0)
#     -> unweighted sine-center
#     -> enforce the theta box on the one common theta
#     -> refit every a_t^(0), d_t^(0) conditional on that common theta
#
# Run this file interactively section by section.

# ============================================================
# 0. Settings
# ============================================================

matrix_part_1 <- "real_data/item_response_matrix.part01.csv"
matrix_part_2 <- "real_data/item_response_matrix.part02.csv"
item_metadata_csv <- "real_data/item_contents.csv"

# Match the intended real-data run here.
theta_bound <- 4.0
a_lower_bound <- 0.01
a_bound <- 4.0
d_bound <- 4.0

# estimate_general_trait defaults used by the Python initialization.
general_lr <- 0.03
general_decay_factor <- 0.5
general_decay_steps <- 200L
general_max_iter <- 1000L
general_tolerance <- 1e-7

# Optional directory for inspection outputs.
output_dir <- "results/estimation/initialization_debug"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)


# ============================================================
# 1. Load the already-cleaned prompt-deduplicated MMLU matrix
# ============================================================
# The current prompt-deduplicated files already have:
#   1. "miscellaneous" removed,
#   2. prompt-dedup column order aligned across metadata and matrix parts.
# So we can skip several of the earlier preprocessing checks.

item_metadata <- read.csv(
  item_metadata_csv,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

part_1 <- read.csv(
  matrix_part_1,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

part_2 <- read.csv(
  matrix_part_2,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

stopifnot("subject" %in% names(item_metadata))
stopifnot(names(part_1)[1] == "model_id")
stopifnot(names(part_2)[1] == "model_id")
stopifnot(identical(names(part_1), names(part_2)))
stopifnot(!any(tolower(trimws(as.character(item_metadata$subject))) == "miscellaneous"))
stopifnot(nrow(item_metadata) == ncol(part_1) - 1L)

item_column_names <- names(part_1)[-1]

model_ids <- c(as.character(part_1$model_id), as.character(part_2$model_id))
stopifnot(!anyDuplicated(model_ids))

Y_all <- rbind(
  as.matrix(part_1[, -1, drop = FALSE]),
  as.matrix(part_2[, -1, drop = FALSE])
)

storage.mode(Y_all) <- "double"
stopifnot(all(Y_all %in% c(0, 1)))

N <- nrow(Y_all)
subject_per_item <- as.character(item_metadata$subject)
subject_names_raw <- unique(subject_per_item)

N
length(subject_names_raw)
dim(Y_all)


# ============================================================
# 2. Split into subject blocks and remove full-data constant items
# ============================================================
# The data files already exclude "miscellaneous", so the only filtering
# left here is the same constant-item removal used by the real-data driver.

Y_blocks <- list()
item_names_blocks <- list()
subject_names <- gsub(" ", "_", subject_names_raw, fixed = TRUE)

for (t in seq_along(subject_names_raw)) {
  subject_raw <- subject_names_raw[t]
  item_pos <- which(subject_per_item == subject_raw)
  Y_t <- Y_all[, item_pos, drop = FALSE]
  item_names_t <- item_column_names[item_pos]

  col_sum_t <- colSums(Y_t)
  keep_t <- (col_sum_t > 0) & (col_sum_t < N)

  Y_t <- Y_t[, keep_t, drop = FALSE]
  item_names_t <- item_names_t[keep_t]

  if (ncol(Y_t) == 0L) {
    stop(paste("No nonconstant items remain in subject", subject_names[t]))
  }

  Y_blocks[[t]] <- Y_t
  item_names_blocks[[t]] <- item_names_t
}

T <- length(Y_blocks)
J_vec <- vapply(Y_blocks, ncol, integer(1))

T
N
J_vec
sum(J_vec)


# ============================================================
# 3. Observation masks
# ============================================================
# Full-data initialization: every retained entry is observed.
# For a CV initialization, replace these masks by the TRAINING masks only.

mask_blocks <- lapply(
  Y_blocks,
  function(Y_t) matrix(TRUE, nrow = nrow(Y_t), ncol = ncol(Y_t))
)


# ============================================================
# 4. Direct subject accuracy -> theta_t^(0)
# ============================================================
# For each subject t:
#   score_it = sum_j m_itj y_itj / sum_j m_itj
# and then project the N-vector onto
#   mean(theta_t)=0, ||theta_t||_2=sqrt(N), |theta_it|<=theta_bound.

accuracy_init <- matrix(NA_real_, nrow = T, ncol = N)
theta_init <- matrix(NA_real_, nrow = T, ncol = N)

for (t in seq_len(T)) {
  Y_t <- Y_blocks[[t]]
  mask_t <- mask_blocks[[t]]

  row_counts <- rowSums(mask_t)
  if (any(row_counts == 0L)) {
    stop(paste("A model has zero observed initialization items in subject", subject_names[t]))
  }

  score_t <- rowSums(mask_t * Y_t) / row_counts
  accuracy_init[t, ] <- score_t

  if (max(score_t) - min(score_t) <= 1e-14) {
    stop(paste("All models have the same observed accuracy in subject", subject_names[t]))
  }

  # First try the ordinary centered-sphere projection.
  raw_vector <- score_t
  centered <- raw_vector - mean(raw_vector)
  centered_norm <- sqrt(sum(centered^2))

  if (centered_norm < 1e-14) {
    stop(paste("Degenerate centered score vector in subject", subject_names[t]))
  }

  projected <- sqrt(N) * centered / centered_norm

  # If it already satisfies the box, this is theta_t^(0).
  if (max(abs(projected)) <= theta_bound + 1e-12) {
    theta_t <- projected
  } else {
    # Exact centered + sphere + box projection used in the Python code.
    minimum_feasible_bound <- if (N %% 2L == 0L) {
      1.0
    } else {
      sqrt(N / (N - 1.0))
    }

    if (theta_bound < minimum_feasible_bound - 1e-12) {
      stop("theta_bound is too small for the centered sphere constraint")
    }

    target_norm <- sqrt(N)
    lower_scale <- 0.0
    upper_scale <- 1.0

    # Evaluate centered clipped vector at upper_scale.
    shift_lower <- min(raw_vector) - theta_bound / upper_scale
    shift_upper <- max(raw_vector) + theta_bound / upper_scale

    for (k in seq_len(70L)) {
      shift <- 0.5 * (shift_lower + shift_upper)
      candidate <- pmin(
        pmax(upper_scale * (raw_vector - shift), -theta_bound),
        theta_bound
      )

      if (sum(candidate) > 0.0) {
        shift_lower <- shift
      } else {
        shift_upper <- shift
      }
    }

    shift <- 0.5 * (shift_lower + shift_upper)
    candidate <- pmin(
      pmax(upper_scale * (raw_vector - shift), -theta_bound),
      theta_bound
    )

    while (sqrt(sum(candidate^2)) < target_norm) {
      upper_scale <- upper_scale * 2.0

      if (upper_scale > 1e16) {
        stop("Could not enforce theta constraints")
      }

      shift_lower <- min(raw_vector) - theta_bound / upper_scale
      shift_upper <- max(raw_vector) + theta_bound / upper_scale

      for (k in seq_len(70L)) {
        shift <- 0.5 * (shift_lower + shift_upper)
        candidate <- pmin(
          pmax(upper_scale * (raw_vector - shift), -theta_bound),
          theta_bound
        )

        if (sum(candidate) > 0.0) {
          shift_lower <- shift
        } else {
          shift_upper <- shift
        }
      }

      shift <- 0.5 * (shift_lower + shift_upper)
      candidate <- pmin(
        pmax(upper_scale * (raw_vector - shift), -theta_bound),
        theta_bound
      )
    }

    for (outer in seq_len(70L)) {
      middle_scale <- 0.5 * (lower_scale + upper_scale)

      shift_lower <- min(raw_vector) - theta_bound / middle_scale
      shift_upper <- max(raw_vector) + theta_bound / middle_scale

      for (k in seq_len(70L)) {
        shift <- 0.5 * (shift_lower + shift_upper)
        candidate <- pmin(
          pmax(middle_scale * (raw_vector - shift), -theta_bound),
          theta_bound
        )

        if (sum(candidate) > 0.0) {
          shift_lower <- shift
        } else {
          shift_upper <- shift
        }
      }

      shift <- 0.5 * (shift_lower + shift_upper)
      candidate <- pmin(
        pmax(middle_scale * (raw_vector - shift), -theta_bound),
        theta_bound
      )

      if (sqrt(sum(candidate^2)) < target_norm) {
        lower_scale <- middle_scale
      } else {
        upper_scale <- middle_scale
      }
    }

    # Final centered clipped vector at upper_scale.
    shift_lower <- min(raw_vector) - theta_bound / upper_scale
    shift_upper <- max(raw_vector) + theta_bound / upper_scale

    for (k in seq_len(70L)) {
      shift <- 0.5 * (shift_lower + shift_upper)
      candidate <- pmin(
        pmax(upper_scale * (raw_vector - shift), -theta_bound),
        theta_bound
      )

      if (sum(candidate) > 0.0) {
        shift_lower <- shift
      } else {
        shift_upper <- shift
      }
    }

    shift <- 0.5 * (shift_lower + shift_upper)
    theta_t <- pmin(
      pmax(upper_scale * (raw_vector - shift), -theta_bound),
      theta_bound
    )
  }

  theta_init[t, ] <- theta_t

  cat(
    "subject", t, subject_names[t],
    "J=", ncol(Y_t),
    "mean(theta0)=", mean(theta_t),
    "norm(theta0)=", sqrt(sum(theta_t^2)),
    "max|theta0|=", max(abs(theta_t)),
    "\n"
  )
}

rownames(theta_init) <- subject_names
rownames(accuracy_init) <- subject_names
colnames(theta_init) <- model_ids
colnames(accuracy_init) <- model_ids

# Constraint checks.
apply(theta_init, 1, mean)
sqrt(rowSums(theta_init^2))
apply(abs(theta_init), 1, max)

# This should be 1 (up to ties): direct accuracy and theta start preserve order.
rank_correlations <- vapply(
  seq_len(T),
  function(t) suppressWarnings(cor(accuracy_init[t, ], theta_init[t, ], method = "spearman")),
  numeric(1)
)
summary(rank_correlations)


# ============================================================
# 5. Given theta_t^(0), fit itemwise logistic regressions
# ============================================================
# For every subject t and item j, fit the ordinary logistic regression
#   logit P(Y_itj = 1) = a_tj * theta_it^(0) + d_tj
# and use the glm coefficients directly as the initialization.

a_init <- vector("list", T)
d_init <- vector("list", T)

for (t in seq_len(T)) {
  Y_t <- Y_blocks[[t]]
  mask_t <- mask_blocks[[t]]
  theta_t <- theta_init[t, ]
  J_t <- ncol(Y_t)

  a_t <- numeric(J_t)
  d_t <- numeric(J_t)

  for (j in seq_len(J_t)) {
    observed <- mask_t[, j]

    if (!any(observed)) {
      a_t[j] <- 0.0
      d_t[j] <- 0.0
      next
    }

    y_obs <- Y_t[observed, j]
    theta_obs <- theta_t[observed]

    glm_fit <- suppressWarnings(
      glm(
        y_obs ~ theta_obs,
        family = binomial(link = "logit")
      )
    )

    coef_glm <- coef(glm_fit)

    d_t[j] <- coef_glm[1]
    a_t[j] <- coef_glm[2]
  }

  a_init[[t]] <- a_t
  d_init[[t]] <- d_t

  cat(
    "subject", t, subject_names[t],
    "initial a range =", range(a_t),
    "initial d range =", range(d_t),
    "\n"
  )
}

names(a_init) <- subject_names
names(d_init) <- subject_names

glm_init_summary <- data.frame(
  J = sapply(a_init, length),
  mean_a = sapply(a_init, function(x) mean(x, na.rm = TRUE)),
  mean_d = sapply(d_init, function(x) mean(x, na.rm = TRUE))
)

glm_init_summary


# ============================================================
# Plot raw-GLM item response curves by subject
# ============================================================

library(ggplot2)

plot_dir <- "glm_raw_item_curves_by_subject"

dir.create(
  plot_dir,
  showWarnings = FALSE,
  recursive = TRUE
)


for (t in seq_along(subject_names)) {
  
  subject_t <- subject_names[t]
  
  theta_t <- theta_init[t, ]
  
  a_raw_t <- a_init[[t]]
  d_raw_t <- d_init[[t]]
  
  J_t <- length(a_raw_t)
  
  
  # ----------------------------------------------------------
  # Use a common theta grid covering this subject's theta range
  # ----------------------------------------------------------
  
  theta_grid <- seq(
    min(theta_t),
    max(theta_t),
    length.out = 300
  )
  
  
  # ----------------------------------------------------------
  # Construct one ICC for every item
  # ----------------------------------------------------------
  
  plot_list <- vector("list", J_t)
  
  for (j in seq_len(J_t)) {
    
    a_j <- a_raw_t[j]
    d_j <- d_raw_t[j]
    
    if (!is.finite(a_j) || !is.finite(d_j)) {
      next
    }
    
    eta_j <- d_j + a_j * theta_grid
    
    p_j <- plogis(eta_j)
    
    plot_list[[j]] <- data.frame(
      theta = theta_grid,
      probability = p_j,
      item = j,
      a = a_j,
      d = d_j
    )
  }
  
  
  plot_list <- plot_list[
    !sapply(plot_list, is.null)
  ]
  
  plot_dat <- do.call(
    rbind,
    plot_list
  )
  
  
  # ----------------------------------------------------------
  # Plot
  # ----------------------------------------------------------
  
  p <- ggplot(
    plot_dat,
    aes(
      x = theta,
      y = probability,
      group = item
    )
  ) +
    geom_line(
      alpha = 0.18,
      linewidth = 0.35
    ) +
    geom_vline(
      xintercept = 0,
      linetype = "dashed",
      linewidth = 0.4
    ) +
    labs(
      title = paste0(
        subject_t,
        ": raw GLM item response curves"
      ),
      subtitle = paste0(
        "J = ", J_t,
        "; coefficients taken directly from glm"
      ),
      x = expression(theta),
      y = expression(
        P(Y == 1 ~ "|" ~ theta)
      )
    ) +
    coord_cartesian(
      ylim = c(0, 1)
    ) +
    theme_bw(base_size = 12)
  
  
  # ----------------------------------------------------------
  # Save
  # ----------------------------------------------------------
  
  file_name <- paste0(
    sprintf("%02d", t),
    "_",
    subject_t,
    "_glm_raw_ICC.png"
  )
  
  ggsave(
    filename = file.path(
      plot_dir,
      file_name
    ),
    plot = p,
    width = 8,
    height = 6,
    dpi = 300
  )
  
  
  cat(
    "saved:",
    file.path(plot_dir, file_name),
    "\n"
  )
}


# ============================================================
# 6. SVD start for the general trait optimizer
# ============================================================
# This SVD vector is only the numerical starting point.
# The actual theta_G^(0) below is the minimizer found for
#   sum_t sin(theta_G, theta_t^(0)).

sv <- svd(theta_init, nu = 0L, nv = 1L)
raw_theta_G <- sv$v[, 1]

theta_G <- raw_theta_G - mean(raw_theta_G)
theta_G <- sqrt(N) * theta_G / sqrt(sum(theta_G^2))

# Orientation constraint: theta_G' sum_t theta_t >= 0.
if (sum(theta_G * colSums(theta_init)) < 0.0) {
  theta_G <- -theta_G
}

mean(theta_G)
sqrt(sum(theta_G^2))
sum(theta_G * colSums(theta_init))


# ============================================================
# 7. Optimize theta_G^(0) = argmin sum_t sin(theta_G, theta_t^(0))
#    using the same projected Adam logic
# ============================================================

first_moment <- rep(0.0, N)
second_moment <- rep(0.0, N)

best_objective <- Inf
best_theta_G <- theta_G
previous_objective <- Inf
converged_G <- FALSE

for (iteration in seq_len(general_max_iter)) {
  norm_G <- sqrt(sum(theta_G^2))
  norms_theta <- sqrt(rowSums(theta_init^2))

  unit_G <- theta_G / norm_G
  unit_theta <- theta_init / norms_theta

  cosine <- as.vector(unit_theta %*% unit_G)
  cosine <- pmin(pmax(cosine, -1.0), 1.0)
  sine <- sqrt(pmax(0.0, 1.0 - cosine^2))

  objective_G <- sum(sine)

  multiplier <- rep(0.0, T)
  nonzero <- sine > 1e-12
  multiplier[nonzero] <- -cosine[nonzero] / sine[nonzero]

  difference_matrix <- unit_theta - cosine * matrix(
    unit_G,
    nrow = T,
    ncol = N,
    byrow = TRUE
  )

  gradient_G <- colSums(multiplier * difference_matrix) / norm_G

  if (objective_G < best_objective) {
    best_objective <- objective_G
    best_theta_G <- theta_G
  }

  if (objective_G <= general_tolerance) {
    converged_G <- TRUE
    break
  }

  relative_change <- if (is.finite(previous_objective)) {
    abs(objective_G - previous_objective) / max(1.0, abs(previous_objective))
  } else {
    Inf
  }

  if (relative_change < general_tolerance) {
    converged_G <- TRUE
    break
  }

  previous_objective <- objective_G

  decay_exponent <- floor((iteration - 1L) / general_decay_steps)
  current_lr <- general_lr * general_decay_factor^decay_exponent

  # Python estimate_general_trait divides this gradient by T before Adam.
  adam_gradient <- gradient_G / T

  first_moment <- 0.9 * first_moment + 0.1 * adam_gradient
  second_moment <- 0.999 * second_moment + 0.001 * adam_gradient^2

  first_unbiased <- first_moment / (1.0 - 0.9^iteration)
  second_unbiased <- second_moment / (1.0 - 0.999^iteration)

  theta_G <- theta_G - current_lr * first_unbiased / (sqrt(second_unbiased) + 1e-8)

  # Project only to the centered sphere here, not to theta_bound.
  theta_G <- theta_G - mean(theta_G)
  theta_G <- sqrt(N) * theta_G / sqrt(sum(theta_G^2))

  if (sum(theta_G * colSums(theta_init)) < 0.0) {
    theta_G <- -theta_G
  }
}

# Python returns the best point seen along the trajectory.
theta_G_init <- best_theta_G

if (sum(theta_G_init * colSums(theta_init)) < 0.0) {
  theta_G_init <- -theta_G_init
}

cat("theta_G initialization iterations =", iteration, "\n")
cat("theta_G initialization converged =", converged_G, "\n")
cat("sum sine at returned theta_G =", best_objective, "\n")
cat("mean(theta_G) =", mean(theta_G_init), "\n")
cat("norm(theta_G) =", sqrt(sum(theta_G_init^2)), "\n")
cat("orientation inner product =", sum(theta_G_init * colSums(theta_init)), "\n")

names(theta_G_init) <- model_ids


# ============================================================
# 8. Proposed initialization objects are now complete
# ============================================================

# Inspect one subject.
subject_to_inspect <- 1L

head(data.frame(
  subject = subject_names[subject_to_inspect],
  accuracy = accuracy_init[subject_to_inspect, ],
  theta_t_0 = theta_init[subject_to_inspect, ],
  theta_G_0 = theta_G_init
))

head(data.frame(
  item_name = item_names_blocks[[subject_to_inspect]],
  a_0 = a_init[[subject_to_inspect]],
  d_0 = d_init[[subject_to_inspect]]
))


# Spearman correlation between theta_t^(0) and theta_G^(0)

spearman_theta_G <- numeric(length(subject_names))

for (t in seq_along(subject_names)) {
  
  spearman_theta_G[t] <- cor(
    theta_init[t, ],
    theta_G,
    method = "spearman"
  )
  
  cat(
    "subject", t,
    subject_names[t],
    "Spearman =", round(spearman_theta_G[t], 4),
    "\n"
  )
}
spearman_theta_G_summary <- data.frame(
  subject = subject_names,
  J = sapply(Y_blocks, ncol),
  spearman_theta_t0_theta_G0 = spearman_theta_G
)

spearman_theta_G_summary[
  order(-spearman_theta_G_summary$spearman_theta_t0_theta_G0),
]


# ============================================================
# PCA of subject-specific initial theta_t^(0)
# rows = subjects
# columns = models
# ============================================================

dim(theta_init)
# should be 56 x N


pca_theta <- prcomp(
  theta_init,
  center = TRUE,
  scale. = FALSE
)

summary(pca_theta)

library(ggplot2)

pca_subject_df <- data.frame(
  subject = subject_names,
  PC1 = pca_theta$x[, 1],
  PC2 = pca_theta$x[, 2]
)

var_explained <- pca_theta$sdev^2 / sum(pca_theta$sdev^2)

ggplot(
  pca_subject_df,
  aes(
    x = PC1,
    y = PC2,
    label = subject
  )
) +
  geom_point(size = 2.5) +
  geom_text(
    size = 3,
    vjust = -0.5,
    check_overlap = TRUE
  ) +
  labs(
    title = expression("PCA of subject-specific " * theta[t]^{(0)}),
    x = paste0(
      "PC1 (",
      round(100 * var_explained[1], 1),
      "%)"
    ),
    y = paste0(
      "PC2 (",
      round(100 * var_explained[2], 1),
      "%)"
    )
  ) +
  theme_bw(base_size = 12)
