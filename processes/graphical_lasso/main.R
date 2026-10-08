# Estimate normative FDG-PET connectivity with graphical lasso.

# Configuration (paths are relative to the repository root)
file_path <- "data/processed/suvr/20251130_suvr.parquet"
output_dir <- "data/processed/graphical_lasso"
lambda_values <- c(
  0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
  0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
)

library(glasso)
library(igraph)
library(Matrix)
library(corrplot)
library(ggplot2)
library(dplyr)
library(reshape2)
library(arrow)

# Load regional SUVR values and group labels from Parquet.
load_suvr_data <- function(file) {
  data <- arrow::read_parquet(file)

  if ("file" %in% colnames(data)) {
    data <- data %>% select(-file)
  }

  suvr_cols <- grep("suvr$", colnames(data), value = TRUE)
  suvr <- as.matrix(data[, suvr_cols])
  groups <- data$group

  cat(sprintf(
    "Data shape: %d subjects x %d regions\n", nrow(suvr), ncol(suvr)
  ))

  return(list(data = data, suvr = suvr, groups = groups))
}

# Standardize control SUVRs after excluding cerebellum and vermis.
filter_and_preprocess <- function(loaded_data) {
  data <- loaded_data$data
  suvr <- loaded_data$suvr
  groups <- loaded_data$groups

  control_mask <- groups %in% c("controls", "controls-adni")
  suvr <- suvr[control_mask, ]
  groups <- groups[control_mask]
  data <- data[control_mask, ]

  cat(sprintf("After filtering controls: %d subjects\n", nrow(suvr)))

  suvr_cols <- colnames(suvr)
  cereb_mask <- !grepl("cereb|vermis", suvr_cols, ignore.case = TRUE)
  suvr <- suvr[, cereb_mask]
  suvr_cols <- suvr_cols[cereb_mask]

  cat(sprintf("After removing cerebellum/vermis: %d regions\n", ncol(suvr)))

  suvr_norm <- scale(suvr, center = TRUE, scale = TRUE)

  return(list(
    data = data,
    suvr = suvr,
    suvr_norm = suvr_norm,
    groups = groups,
    region_names = suvr_cols
  ))
}

# Evaluate regularization parameters with five-fold cross-validation by default.
cv_glasso <- function(data, lambda_values, k_folds = 5) {
  n <- nrow(data)
  p <- ncol(data)

  cat(sprintf(
    "Performing %d-fold cross-validation for %d lambda values...\n",
    k_folds, length(lambda_values)
  ))

  set.seed(2025)
  fold_indices <- sample(rep(1:k_folds, length.out = n))

  cv_scores <- matrix(0, nrow = length(lambda_values), ncol = k_folds)

  for (i in seq_along(lambda_values)) {
    lambda <- lambda_values[i]

    for (fold in 1:k_folds) {
      train_idx <- fold_indices != fold
      test_idx <- fold_indices == fold

      train_data <- data[train_idx, ]
      test_data <- data[test_idx, ]

      S_train <- cov(train_data)
      S_test <- cov(test_data)

      tryCatch({
        glasso_fit <- glasso(
          S_train, rho = lambda, trace = FALSE, maxit = 1000,
          penalize.diagonal = FALSE
        )

        log_lik <- sum(diag(S_test %*% glasso_fit$wi)) -
          log(det(glasso_fit$wi)) - p

        cv_scores[i, fold] <- log_lik
      }, error = function(e) {
        cv_scores[i, fold] <- -Inf
      })
    }

    if (i %% 2 == 0 || i == length(lambda_values)) {
      cat(sprintf("Completed lambda %.3f (%d/%d)\n", lambda, i, length(lambda_values)))
    }
  }

  cv_mean <- rowMeans(cv_scores, na.rm = TRUE)
  cv_se <- apply(cv_scores, 1, sd, na.rm = TRUE) / sqrt(k_folds)

  optimal_idx <- which.min(cv_mean)
  optimal_lambda <- lambda_values[optimal_idx]

  cat(sprintf("Optimal lambda by CV: %.3f\n", optimal_lambda))

  return(list(
    lambda_values = lambda_values,
    cv_mean = cv_mean,
    cv_se = cv_se,
    cv_scores = cv_scores,
    optimal_lambda = optimal_lambda,
    optimal_idx = optimal_idx
  ))
}

# Evaluate AIC, BIC, and EBIC for each regularization parameter.
calculate_model_selection_criteria <- function(data, lambda_values) {
  n <- nrow(data)
  p <- ncol(data)
  S <- cov(data)

  cat("Calculating model selection criteria...\n")

  results <- data.frame(
    lambda = lambda_values,
    aic = numeric(length(lambda_values)),
    bic = numeric(length(lambda_values)),
    ebic = numeric(length(lambda_values)),
    n_edges = numeric(length(lambda_values)),
    log_likelihood = numeric(length(lambda_values)),
    converged = logical(length(lambda_values))
  )

  for (i in seq_along(lambda_values)) {
    lambda <- lambda_values[i]

    tryCatch({
      glasso_fit <- glasso(
        S, rho = lambda, trace = FALSE, penalize.diagonal = FALSE
      )

      precision_binary <- abs(glasso_fit$wi) > 1e-6
      diag(precision_binary) <- FALSE
      n_edges <- sum(precision_binary) / 2

      log_lik <- n / 2 * (log(det(glasso_fit$wi)) - sum(diag(S %*% glasso_fit$wi)))

      aic <- -2 * log_lik + 2 * n_edges
      bic <- -2 * log_lik + log(n) * n_edges

      gamma <- 0.5
      ebic <- bic + 4 * gamma * n_edges * log(p)

      results[i, ] <- c(lambda, aic, bic, ebic, n_edges, log_lik, TRUE)
    }, error = function(e) {
      results[i, ] <- c(lambda, NA, NA, NA, NA, NA, FALSE)
    })
  }

  if (any(!is.na(results$aic))) {
    optimal_aic <- results$lambda[which.min(results$aic)]
    optimal_bic <- results$lambda[which.min(results$bic)]
    optimal_ebic <- results$lambda[which.min(results$ebic)]

    cat(sprintf("Optimal lambda by AIC: %.3f\n", optimal_aic))
    cat(sprintf("Optimal lambda by BIC: %.3f\n", optimal_bic))
    cat(sprintf("Optimal lambda by EBIC: %.3f\n", optimal_ebic))
  }

  return(results)
}

# Build binary normative graphs from nonzero off-diagonal precision entries.
compute_connectivity_networks <- function(data, lambda_values, region_names) {
  n <- nrow(data)
  p <- ncol(data)
  S <- cov(data)

  S <- (S + t(S)) / 2

  cat(sprintf(
    "Computing connectivity networks for %d lambda values...\n",
    length(lambda_values)
  ))

  connectivity_matrices <- list()
  precision_matrices <- list()
  model_info <- list()

  for (i in seq_along(lambda_values)) {
    lambda <- lambda_values[i]
    cat(sprintf("----- Processing lambda: %.3f -----\n", lambda))

    tryCatch({
      glasso_fit <- glasso(
        s = S,
        rho = lambda,
        trace = FALSE,
        maxit = 1000,
        penalize.diagonal = FALSE
      )

      precision_binary <- abs(glasso_fit$wi) > 1e-6
      diag(precision_binary) <- FALSE

      n_edges <- sum(precision_binary) / 2
      cat(sprintf("Number of edges: %.0f\n", n_edges))

      connectivity_df <- as.data.frame(precision_binary * 1)
      rownames(connectivity_df) <- region_names
      colnames(connectivity_df) <- region_names

      g <- graph_from_adjacency_matrix(
        as.matrix(precision_binary), mode = "undirected"
      )
      is_graph_connected <- is_connected(g)

      if (!is_graph_connected) {
        warning(sprintf(
          "Graph associated with lambda %.3f contains unconnected nodes.", lambda
        ))
      } else {
        cat("Graph is connected.\n")
      }

      lambda_key <- as.character(lambda)
      connectivity_matrices[[lambda_key]] <- connectivity_df
      precision_matrices[[lambda_key]] <- glasso_fit$wi

      model_info[[lambda_key]] <- list(
        lambda = lambda,
        n_edges = n_edges,
        is_connected = is_graph_connected,
        converged = TRUE,
        covariance = glasso_fit$w,
        precision = glasso_fit$wi
      )
    }, error = function(e) {
      cat(sprintf("Error processing lambda %.3f: %s\n", lambda, e$message))

      lambda_key <- as.character(lambda)
      model_info[[lambda_key]] <- list(
        lambda = lambda,
        n_edges = NA,
        is_connected = FALSE,
        converged = FALSE,
        error = e$message
      )
    })

    cat("\n")
  }

  cat("Connectivity network computation completed.\n")

  return(list(
    connectivity_matrices = connectivity_matrices,
    precision_matrices = precision_matrices,
    model_info = model_info
  ))
}

# Plot information criteria and edge counts across regularization parameters.
plot_model_selection <- function(criteria_results) {
  plot_data <- criteria_results %>%
    select(lambda, aic, bic, ebic) %>%
    reshape2::melt(
      id.vars = "lambda", variable.name = "criterion", value.name = "value"
    )

  p1 <- ggplot(plot_data, aes(x = lambda, y = value, color = criterion)) +
    geom_line(size = 1) +
    geom_point(size = 2) +
    labs(
      title = "Model Selection Criteria",
      x = "Lambda (Regularization Parameter)",
      y = "Criterion Value",
      color = "Criterion"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom") +
    facet_wrap(~ criterion, scales = "free_y", ncol = 2)

  p2 <- ggplot(criteria_results, aes(x = lambda, y = n_edges)) +
    geom_line(size = 1, color = "darkblue") +
    geom_point(size = 2, color = "darkblue") +
    labs(
      title = "Network Sparsity vs Lambda",
      x = "Lambda (Regularization Parameter)",
      y = "Number of Edges"
    ) +
    theme_minimal()

  p3 <- ggplot(plot_data, aes(x = lambda, y = value, color = criterion)) +
    geom_line(size = 1) +
    geom_point(size = 2) +
    labs(
      title = "Model Selection Criteria Comparison",
      x = "Lambda (Regularization Parameter)",
      y = "Criterion Value (Standardized)",
      color = "Criterion"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom") +
    scale_y_continuous(labels = scales::scientific)

  return(list(
    criteria_facet = p1,
    edges_plot = p2,
    criteria_combined = p3
  ))
}

# Plot cross-validation scores and standard errors.
plot_cv_results <- function(cv_results) {
  cv_data <- data.frame(
    lambda = cv_results$lambda_values,
    cv_mean = cv_results$cv_mean,
    cv_se = cv_results$cv_se
  )

  optimal_lambda <- cv_results$optimal_lambda

  p <- ggplot(cv_data, aes(x = lambda, y = cv_mean)) +
    geom_line(size = 1, color = "red") +
    geom_point(size = 2, color = "red") +
    geom_errorbar(
      aes(ymin = cv_mean - cv_se, ymax = cv_mean + cv_se),
      width = 0.02, color = "red", alpha = 0.7
    ) +
    geom_vline(
      xintercept = optimal_lambda,
      linetype = "dashed",
      color = "blue",
      alpha = 0.7
    ) +
    annotate(
      "text",
      x = optimal_lambda,
      y = max(cv_data$cv_mean, na.rm = TRUE),
      label = sprintf("Optimal λ = %.3f", optimal_lambda),
      vjust = -0.5,
      color = "blue"
    ) +
    labs(
      title = "Cross-Validation for Lambda Selection",
      x = "Lambda (Regularization Parameter)",
      y = "Average negative Log-Likelihood",
      caption = "Error bars represent standard error across folds"
    ) +
    theme_minimal()

  return(p)
}

# Plot the upper triangle of a binary connectivity matrix.
plot_connectivity_matrix <- function(connectivity_matrix,
                                    title = "Connectivity Matrix",
                                    method = "color") {
  if (is.data.frame(connectivity_matrix)) {
    conn_matrix <- as.matrix(connectivity_matrix)
  } else {
    conn_matrix <- connectivity_matrix
  }

  corrplot(
    conn_matrix,
    method = method,
    type = "upper",
    order = "hclust",
    title = title,
    mar = c(0, 0, 1, 0),
    col.lim = c(0, 1),
    is.corr = FALSE,
    tl.cex = 0.7,
    tl.col = "black"
  )
}

# Plot an undirected connectivity graph.
plot_network_graph <- function(connectivity_matrix, title = "Network Graph") {
  g <- graph_from_adjacency_matrix(
    as.matrix(connectivity_matrix), mode = "undirected"
  )

  layout <- layout_with_fr(g)

  plot(
    g,
    layout = layout,
    vertex.size = 8,
    vertex.color = "lightblue",
    vertex.label.cex = 0.7,
    edge.color = "gray",
    main = title
  )

  return(g)
}

# Estimate normative graphs, generate figures, and export connectivity matrices.
main_analysis <- function(file_path,
                          lambda_values = c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                          output_dir = "results") {
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  cat("=== FDG-PET CONNECTIVITY ANALYSIS WITH GRAPHICAL LASSO ===\n\n")

  cat("Step 1: Loading and preprocessing data...\n")
  loaded_data <- load_suvr_data(file_path)
  processed_data <- filter_and_preprocess(loaded_data)

  suvr_norm <- processed_data$suvr_norm
  region_names <- processed_data$region_names

  cat(sprintf(
    "Final input shape: %d subjects x %d regions\n\n",
    nrow(suvr_norm), ncol(suvr_norm)
  ))

  cat("Step 2: Performing cross-validation...\n")
  cv_results <- cv_glasso(suvr_norm, lambda_values, k_folds = 5)
  cat("\n")

  cat("Step 3: Calculating model selection criteria...\n")
  criteria_results <- calculate_model_selection_criteria(suvr_norm, lambda_values)
  cat("\n")

  cat("Step 4: Computing connectivity networks...\n")
  network_results <- compute_connectivity_networks(
    suvr_norm, lambda_values, region_names
  )
  cat("\n")

  cat("Step 5: Generating visualizations...\n")

  selection_plots <- plot_model_selection(criteria_results)
  cv_plot <- plot_cv_results(cv_results)

  ggsave(
    file.path(output_dir, "model_selection_criteria.png"),
    selection_plots$criteria_facet, width = 12, height = 8, dpi = 300
  )
  ggsave(
    file.path(output_dir, "network_sparsity.png"),
    selection_plots$edges_plot, width = 8, height = 6, dpi = 300
  )
  ggsave(
    file.path(output_dir, "cross_validation.png"),
    cv_plot, width = 8, height = 6, dpi = 300
  )

  if (!is.na(criteria_results$ebic[1])) {
    optimal_ebic <- criteria_results$lambda[which.min(criteria_results$ebic)]
    optimal_connectivity <- network_results$connectivity_matrices[[
      as.character(optimal_ebic)
    ]]

    png(
      file.path(output_dir, sprintf("connectivity_matrix_lambda_%.3f.png", optimal_ebic)),
      width = 800, height = 800
    )
    plot_connectivity_matrix(
      optimal_connectivity,
      title = sprintf("Connectivity Matrix (λ = %.3f)", optimal_ebic)
    )
    dev.off()
  }

  cat("Visualizations saved to:", output_dir, "\n\n")

  cat("Step 6: Saving results...\n")

  for (lambda in names(network_results$connectivity_matrices)) {
    conn_matrix <- network_results$connectivity_matrices[[lambda]]

    conn_matrix <- as.data.frame(lapply(conn_matrix, as.integer))
    rownames(conn_matrix) <- region_names
    colnames(conn_matrix) <- region_names

    conn_matrix_export <- conn_matrix
    conn_matrix_export$region <- rownames(conn_matrix_export)
    conn_matrix_export <- conn_matrix_export[, c("region", region_names)]

    arrow::write_parquet(
      conn_matrix_export,
      file.path(output_dir, sprintf("connectivity_matrix_lambda_%s.parquet", lambda))
    )
  }

  cat(
    "Results saved to:",
    file.path(output_dir, "connectivity_analysis_results.RData"), "\n\n"
  )

  cat("=== ANALYSIS SUMMARY ===\n")
  cat(sprintf("Total subjects analyzed: %d\n", nrow(suvr_norm)))
  cat(sprintf("Total brain regions: %d\n", ncol(suvr_norm)))
  cat(sprintf("Lambda values tested: %s\n", paste(lambda_values, collapse = ", ")))

  if (!is.na(cv_results$optimal_lambda)) {
    cat(sprintf(
      "Optimal lambda by Cross-Validation: %.3f\n", cv_results$optimal_lambda
    ))
  }

  if (any(!is.na(criteria_results$aic))) {
    cat(sprintf(
      "Optimal lambda by AIC: %.3f\n",
      criteria_results$lambda[which.min(criteria_results$aic)]
    ))
    cat(sprintf(
      "Optimal lambda by BIC: %.3f\n",
      criteria_results$lambda[which.min(criteria_results$bic)]
    ))
    cat(sprintf(
      "Optimal lambda by EBIC: %.3f\n",
      criteria_results$lambda[which.min(criteria_results$ebic)]
    ))
  }

  cat("\nConnectivity matrices available in: network_results$connectivity_matrices\n")
  cat("Model selection criteria in: criteria_results\n")
  cat("Cross-validation results in: cv_results\n")

  return(list(
    network_results = network_results,
    criteria_results = criteria_results,
    cv_results = cv_results,
    processed_data = processed_data,
    parameters = list(
      lambda_values = lambda_values,
      file_path = file_path,
      output_dir = output_dir
    )
  ))
}

results <- main_analysis(
  file_path = file_path,
  lambda_values = lambda_values,
  output_dir = output_dir
)

connectivity_matrices <- results$network_results$connectivity_matrices
optimal_lambda_cv <- results$cv_results$optimal_lambda
lambda_connectivity <- connectivity_matrices[["0.06"]]

cat("FDG-PET Connectivity Analysis Script Loaded Successfully!\n")
