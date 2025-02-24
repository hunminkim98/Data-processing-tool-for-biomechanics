# Install required packages if not already installed
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("lme4")) install.packages("lme4")
if (!require("BlandAltmanLeh")) install.packages("BlandAltmanLeh")
if (!require("irr")) install.packages("irr")

# Load required libraries
library(tidyverse)
library(ggplot2)
library(lme4)
library(BlandAltmanLeh)
library(irr)
library(readxl)
library(openxlsx)
library(zoo)

# Set the parent directory
parent_dir <- "D:/석사/석사3차/Markerless validation/Results/Final2/merged_check"

# Update find_edited_files function
find_edited_files <- function(motion_type = NULL, subject_type = NULL) {
  # Base directory
  base_dir <- "D:/석사/석사3차/Markerless validation/Results/Final2/merged_check"
  
  # Create pattern based on motion type
  motion_pattern <- if (!is.null(motion_type)) {
    paste0(".*", motion_type, ".*")
  } else {
    ".*"
  }
  
  # Create pattern based on subject type
  subject_pattern <- if (!is.null(subject_type)) {
    paste0(".*", subject_type, ".*")
  } else {
    ".*"
  }
  
  # Find all edited Excel files in the directory and subdirectories
  files <- list.files(
    path = base_dir,
    pattern = "_edited\\.xlsx$",  # Only find files ending with _edited.xlsx
    recursive = TRUE,
    full.names = TRUE
  )
  
  # Filter files based on motion type and subject type
  filtered_files <- files[grepl(motion_pattern, basename(files), ignore.case = TRUE) &
                            grepl(subject_pattern, dirname(files), ignore.case = TRUE)]
  
  # Remove files with "plots" in their path
  filtered_files <- filtered_files[!grepl("plots", filtered_files, ignore.case = TRUE)]
  
  # Convert to data frame
  files_df <- data.frame(
    file_path = filtered_files,
    stringsAsFactors = FALSE
  )
  
  # Print found files for debugging
  cat("\nFound files:\n")
  for (file in filtered_files) {
    cat("  ", basename(file), "\n")
  }
  
  return(files_df)
}

# Function to find peak value
find_peak_value <- function(data) {
    # Find absolute peak value
    abs_data <- abs(data)
    peak_index <- which.max(abs_data)
    return(data[peak_index])
}

# Function to find mean value
find_mean_value <- function(data) {
    return(mean(data, na.rm = TRUE))
}

# Update read_and_process_file function to include peak detection
read_and_process_file <- function(file_path) {
    if (!file.exists(file_path)) {
        warning("File does not exist: ", file_path)
        return(NULL)
    }
    
    tryCatch({
        sheets <- excel_sheets(file_path)
        data <- list()
        
        is_subject_kiHoon <- grepl("성기훈", file_path, fixed = TRUE)
        is_subject_Ryan <- grepl("김리언", file_path, fixed = TRUE)
        is_subject_TH <- grepl("김태형", file_path, fixed = TRUE)
        is_swing_motion <- grepl("swing", file_path, ignore.case = TRUE)
        
        processed_joints <- list()
        
        for (sheet in sheets) {
            # cat("\nChecking sheet:", sheet)
            
            if (grepl("shoulder", sheet, ignore.case = TRUE)) {
                next
            }
            
            # 성기훈의 Left Knee 제외 (swing 동작)
            if (is_subject_kiHoon && is_swing_motion && grepl("knee", sheet, ignore.case = TRUE)) {
                if (!grepl("Right", sheet, ignore.case = TRUE)) {
                    next
                }
            }
            
            # 김리언의 ankle 제외 (swing 동작)
            if (is_subject_Ryan && is_swing_motion && grepl("ankle", sheet, ignore.case = TRUE)) {
                next
            }

            # 김태형의 trunk 제외 (swing 동작)
            if (is_subject_TH && is_swing_motion && grepl("trunk", sheet, ignore.case = TRUE)) {
                next
            }
            
            sheet_data <- read_excel(
                file_path, 
                sheet = sheet,
                col_names = FALSE,
                .name_repair = "minimal"
            )
            
            data_rows <- sheet_data[-(1:3), ]
            
            marker_data <- data.frame(
                Frame = seq_len(nrow(data_rows)),
                X = as.numeric(as.character(data_rows[[2]])),
                Y = as.numeric(as.character(data_rows[[3]])),
                Z = as.numeric(as.character(data_rows[[4]])),
                stringsAsFactors = FALSE
            )
            
            markerless_data <- data.frame(
                Frame = seq_len(nrow(data_rows)),
                X = as.numeric(as.character(data_rows[[6]])),
                Y = as.numeric(as.character(data_rows[[7]])),
                Z = as.numeric(as.character(data_rows[[8]])),
                stringsAsFactors = FALSE
            )
            
            valid_rows <- complete.cases(marker_data[, c("X", "Y", "Z")]) & 
                         complete.cases(markerless_data[, c("X", "Y", "Z")])
            
            marker_data <- marker_data[valid_rows, ]
            markerless_data <- markerless_data[valid_rows, ]
            
            # After processing marker and markerless data, find peak values
            marker_peaks <- list(
                X = find_peak_value(marker_data$X),
                Y = find_peak_value(marker_data$Y),
                Z = find_peak_value(marker_data$Z)
            )
            
            markerless_peaks <- list(
                X = find_peak_value(markerless_data$X),
                Y = find_peak_value(markerless_data$Y),
                Z = find_peak_value(markerless_data$Z)
            )
            
            base_joint_name <- gsub("^(Left|Right)_", "", sheet)
            
            if (!(base_joint_name %in% names(processed_joints))) {
                processed_joints[[base_joint_name]] <- list(
                    marker_based = marker_data,
                    markerless = markerless_data,
                    marker_peaks = marker_peaks,
                    markerless_peaks = markerless_peaks
                )
            } else {
                existing <- processed_joints[[base_joint_name]]
                common_frames <- intersect(marker_data$Frame, existing$marker_based$Frame)
                
                if (length(common_frames) > 0) {
                    marker_subset <- marker_data[marker_data$Frame %in% common_frames, ]
                    markerless_subset <- markerless_data[markerless_data$Frame %in% common_frames, ]
                    existing_marker <- existing$marker_based[existing$marker_based$Frame %in% common_frames, ]
                    existing_markerless <- existing$markerless[existing$markerless$Frame %in% common_frames, ]
                    
                    processed_joints[[base_joint_name]] <- list(
                        marker_based = data.frame(
                            Frame = common_frames,
                            X = (existing_marker$X + marker_subset$X) / 2,
                            Y = (existing_marker$Y + marker_subset$Y) / 2,
                            Z = (existing_marker$Z + marker_subset$Z) / 2
                        ),
                        markerless = data.frame(
                            Frame = common_frames,
                            X = (existing_markerless$X + markerless_subset$X) / 2,
                            Y = (existing_markerless$Y + markerless_subset$Y) / 2,
                            Z = (existing_markerless$Z + markerless_subset$Z) / 2
                        ),
                        marker_peaks = list(
                            X = max(abs(c(existing$marker_peaks$X, marker_peaks$X))) * sign(marker_peaks$X),
                            Y = max(abs(c(existing$marker_peaks$Y, marker_peaks$Y))) * sign(marker_peaks$Y),
                            Z = max(abs(c(existing$marker_peaks$Z, marker_peaks$Z))) * sign(marker_peaks$Z)
                        ),
                        markerless_peaks = list(
                            X = max(abs(c(existing$markerless_peaks$X, markerless_peaks$X))) * sign(markerless_peaks$X),
                            Y = max(abs(c(existing$markerless_peaks$Y, markerless_peaks$Y))) * sign(markerless_peaks$Y),
                            Z = max(abs(c(existing$markerless_peaks$Z, markerless_peaks$Z))) * sign(markerless_peaks$Z)
                        )
                    )
                }
            }
        }
        
        for (joint_name in names(processed_joints)) {
            data[[joint_name]] <- list(
                marker_based = processed_joints[[joint_name]]$marker_based,
                markerless = processed_joints[[joint_name]]$markerless,
                marker_peaks = processed_joints[[joint_name]]$marker_peaks,
                markerless_peaks = processed_joints[[joint_name]]$markerless_peaks
            )
        }
        
        return(data)
    }, error = function(e) {
        warning(paste("Error processing file:", file_path, "\nError:", e$message))
        return(NULL)
    })
}

# Update synchronize_data function to handle frame alignment and user approval
synchronize_data <- function(marker_data, markerless_data, joint_name) {
  # Ensure Frame columns are numeric
  marker_data$Frame <- as.numeric(marker_data$Frame)
  markerless_data$Frame <- as.numeric(markerless_data$Frame)
  
  # Remove any NA values
  marker_data <- marker_data %>% filter(!is.na(Frame))
  markerless_data <- markerless_data %>% filter(!is.na(Frame))
  
  # Find the overlapping frame range
  start_frame <- max(min(marker_data$Frame), min(markerless_data$Frame))
  end_frame <- min(max(marker_data$Frame), max(markerless_data$Frame))
  
  cat("\nFrame range:", start_frame, "to", end_frame)
  
  # Filter data to overlapping frames
  marker_subset <- marker_data %>% 
    filter(Frame >= start_frame & Frame <= end_frame) %>%
    arrange(Frame)
  
  markerless_subset <- markerless_data %>% 
    filter(Frame >= start_frame & Frame <= end_frame) %>%
    arrange(Frame)
  
  # Ensure equal number of frames
  min_length <- min(nrow(marker_subset), nrow(markerless_subset))
  if (min_length > 0) {
    marker_subset <- marker_subset[1:min_length,]
    markerless_subset <- markerless_subset[1:min_length,]
    
    # Create new plotting window
    dev.new(width=10, height=8)
    
    # Set up plotting parameters
    old_par <- par(no.readonly = TRUE)  # Save old parameters
    par(mfrow=c(3,1), mar=c(4,4,2,1))
    
    # X coordinate plot
    plot(marker_subset$Frame, marker_subset$X, type="l", col="blue", 
         main=paste(joint_name, "- X Coordinate"), xlab="Frame", ylab="X Position (mm)")
    lines(markerless_subset$Frame, markerless_subset$X, col="red")
    legend("topright", legend=c("Marker", "Markerless"), col=c("blue", "red"), lty=1)
    grid()
    
    # Y coordinate plot
    plot(marker_subset$Frame, marker_subset$Y, type="l", col="blue",
         main="Y Coordinate", xlab="Frame", ylab="Y Position (mm)")
    lines(markerless_subset$Frame, markerless_subset$Y, col="red")
    grid()
    
    # Z coordinate plot
    plot(marker_subset$Frame, marker_subset$Z, type="l", col="blue",
         main="Z Coordinate", xlab="Frame", ylab="Z Position (mm)")
    lines(markerless_subset$Frame, markerless_subset$Z, col="red")
    grid()
    
    # Ask for user approval
    user_response <- readline(prompt=paste("\nAccept synchronization for", joint_name, "? (y/n): "))
    
    # Restore old plotting parameters
    par(old_par)
    dev.off()  # Close the plotting window
    
    if (tolower(user_response) == "y") {
      cat("\nSynchronization accepted for", joint_name, "-", min_length, "frames")
      return(list(
        marker_based = marker_subset,
        markerless = markerless_subset,
        accepted = TRUE
      ))
    } else {
      cat("\nSynchronization rejected for", joint_name)
      return(list(
        marker_based = NULL,
        markerless = NULL,
        accepted = FALSE
      ))
    }
  } else {
    cat("\nNo overlapping frames found")
    return(list(
      marker_based = NULL,
      markerless = NULL,
      accepted = FALSE
    ))
  }
}


# Function to process data from one sheet
process_sheet_data <- function(data) {
  # Skip the first two rows (header information)
  data <- data %>% slice(3:n())
  
  # Split into marker-based and markerless data
  marker_based <- data %>%
    select(1:4) %>%
    rename(
      Frame = 1,
      X = 2,
      Y = 3,
      Z = 4
    ) %>%
    mutate(across(c(X, Y, Z), as.numeric))
  
  markerless <- data %>%
    select(5:8) %>%
    rename(
      Frame = 1,
      X = 2,
      Y = 3,
      Z = 4
    ) %>%
    mutate(across(c(X, Y, Z), as.numeric))
  
  # Synchronize the data
  synchronized_data <- synchronize_data(marker_based, markerless, "joint")
  
  return(synchronized_data)
}

# Function to read all edited Excel files
read_all_files <- function(file_info) {
  # Initialize empty list to store data
  all_data <- list()
  
  # Read each file
  for(i in 1:nrow(file_info)) {
    tryCatch({
      # Read the Excel file
      data <- read_excel(file_info$file_path[i])
      
      # Add metadata
      data$subject <- file_info$subject[i]
      data$motion <- file_info$motion[i]
      data$file_name <- basename(file_info$file_path[i])
      
      # Store in list
      all_data[[i]] <- data
    }, error = function(e) {
      warning(sprintf("Error reading file %s: %s", file_info$file_path[i], e$message))
    })
  }
  
  # Combine all data frames
  combined_data <- bind_rows(all_data)
  return(combined_data)
}

# Data Analysis Plan:
# 1. Data Import and Preprocessing
# 2. Descriptive Statistics
# 3. Correlation Analysis
# 4. Bland-Altman Analysis
# 5. Mixed Effects Model
# 6. Visualization

# Function to calculate basic statistics
calculate_stats <- function(data) {
  stats <- data %>%
    summarise(
      mean = mean(difference),
      sd = sd(difference),
      rmse = sqrt(mean(difference^2)),
      mae = mean(abs(difference))
    )
  return(stats)
}

# Function for correlation analysis
correlation_analysis <- function(markerless, markerbased) {
  cor_test <- cor.test(markerless, markerbased, method = "pearson")
  return(cor_test)
}

# Function for Bland-Altman analysis
bland_altman_analysis <- function(data, joint_name = "", axis_name = "") {
  differences <- data$marker_value - data$markerless_value
  means <- (data$marker_value + data$markerless_value) / 2
  
  mean_diff <- mean(differences, na.rm = TRUE)
  sd_diff <- sd(differences, na.rm = TRUE)
  
  upper_loa <- mean_diff + 1.96 * sd_diff
  lower_loa <- mean_diff - 1.96 * sd_diff
  
  y_min <- min(differences, na.rm = TRUE) - 20
  y_max <- max(differences, na.rm = TRUE) + 20
  
  lines_df <- data.frame(
    yintercept = c(mean_diff, upper_loa, lower_loa),
    Line = factor(c("Mean difference", "Upper LOA", "Lower LOA"),
                 levels = c("Mean difference", "Upper LOA", "Lower LOA"))
  )
  
  stats_labels <- c(
    sprintf("Mean diff: %.2f°", mean_diff),
    sprintf("Upper LOA: %.2f°", upper_loa),
    sprintf("Lower LOA: %.2f°", lower_loa)
  )
  
  custom_theme <- theme_minimal() +
    theme(
      text = element_text(size = 16, color = "black"),
      axis.title = element_text(size = 18, face = "bold", color = "black"),
      axis.text = element_text(size = 24, color = "black"),
      legend.title = element_blank(),
      legend.text = element_text(size = 14, color = "black"),
      legend.position = c(0.95, 0.95),
      legend.justification = c("right", "top"),
      legend.box.background = element_rect(color = "black", fill = "white"),
      legend.box.margin = margin(8, 8, 8, 8),
      plot.title = element_text(size = 20, face = "bold", hjust = 0.5, color = "black"),
      panel.background = element_rect(fill = "white", color = "black"),
      plot.background = element_rect(fill = "white"),
      panel.grid.major = element_line(color = "grey90"),
      panel.grid.minor = element_line(color = "grey95")
    )
  
  plot_title <- sprintf("%s %s-axis", joint_name, axis_name)
  
  plot <- ggplot(data.frame(means = means, differences = differences), aes(x = means, y = differences)) +
    geom_point(alpha = 0.5, color = "black") +
    geom_hline(data = lines_df, aes(yintercept = yintercept, color = Line, linetype = Line)) +
    geom_hline(yintercept = 0, color = "grey50", linewidth = 0.5) +
    scale_color_manual(values = c("Mean difference" = "blue", "Upper LOA" = "red", "Lower LOA" = "red"),
                      labels = stats_labels) +
    scale_linetype_manual(values = c("Mean difference" = "solid", "Upper LOA" = "dashed", "Lower LOA" = "dashed"),
                         labels = stats_labels) +
    labs(
      title = NULL,
      x = NULL,
      y = NULL
    ) +
    custom_theme +
    coord_cartesian(ylim = c(-40, 40)) +
    scale_y_continuous(breaks = seq(-40, 40, by = 20))
  
  results <- list(
    plot = plot,
    statistics = list(
      mean_difference = mean_diff,
      sd_difference = sd_diff,
      upper_loa = upper_loa,
      lower_loa = lower_loa
    )
  )
  
  return(results)
}

# Function for mixed effects model
mixed_model_analysis <- function(data) {
  # Example mixed model
  # model <- lmer(difference ~ (1|subject), data = data)
  # return(summary(model))
}

# Main analysis pipeline will be implemented here
# after data structure is confirmed

# Function to calculate RMSE between marker-based and markerless data
calculate_rmse <- function(all_marker_data, all_markerless_data) {
  calculate_rmse_single <- function(data_list1, data_list2) {
    # Combine all trials
    all_data1 <- do.call(rbind, data_list1)
    all_data2 <- do.call(rbind, data_list2)
    
    # Calculate RMSE using all data points
    rmse <- sqrt(mean((all_data1 - all_data2)^2, na.rm = TRUE))
    return(rmse)
  }
  
  # Prepare data lists for each coordinate
  marker_x_list <- lapply(all_marker_data, function(d) d$X)
  marker_y_list <- lapply(all_marker_data, function(d) d$Y)
  marker_z_list <- lapply(all_marker_data, function(d) d$Z)
  
  markerless_x_list <- lapply(all_markerless_data, function(d) d$X)
  markerless_y_list <- lapply(all_markerless_data, function(d) d$Y)
  markerless_z_list <- lapply(all_markerless_data, function(d) d$Z)
  
  # Calculate RMSE for each coordinate using all trials
  rmse_x <- calculate_rmse_single(marker_x_list, markerless_x_list)
  rmse_y <- calculate_rmse_single(marker_y_list, markerless_y_list)
  rmse_z <- calculate_rmse_single(marker_z_list, markerless_z_list)
  
  return(list(
    X = rmse_x,
    Y = rmse_y,
    Z = rmse_z,
    Total = sqrt(mean(c(rmse_x^2, rmse_y^2, rmse_z^2)))
  ))
}

# Function to calculate ICC between marker-based and markerless data
calculate_icc <- function(reference, comparison) {
  # Remove NA values
  valid_indices <- !is.na(reference) & !is.na(comparison)
  reference <- reference[valid_indices]
  comparison <- comparison[valid_indices]
  
  # Create a data frame where each column represents a measurement method
  data_matrix <- data.frame(reference = reference, comparison = comparison)
  
  # Calculate ICC using irr package
  icc_result <- irr::icc(data_matrix,
                         model = "twoway",    # Two-way Mixed Effects Model
                         type = "consistency",  # Consistency
                         unit = "average")     # Single measurement basis
  
  # Return both ICC value and p-value
  return(list(
    value = icc_result$value,
    p.value = icc_result$p.value
  ))
}

# Function to calculate RMSE between marker-based and markerless data
process_coordinate <- function(marker_data, markerless_data, joint, coord) {
  marker_values <- marker_data[[coord]]
  markerless_values <- markerless_data[[coord]]
  
  if (is.null(marker_values) || is.null(markerless_values) ||
      length(marker_values) == 0 || length(markerless_values) == 0) {
    cat("\nMissing or invalid data for", joint, "-", coord)
    return(NULL)
  }
  
  valid_indices <- !is.na(marker_values) & !is.na(markerless_values)
  marker_values <- marker_values[valid_indices]
  markerless_values <- markerless_values[valid_indices]
  
  if (length(marker_values) < 2 || length(markerless_values) < 2) {
    cat("\nNot enough valid data points for", joint, "-", coord)
    return(NULL)
  }
  
  # RMSE 계산
  rmse <- sqrt(mean((marker_values - markerless_values)^2, na.rm = TRUE))
  
  # ICC 계산
  icc_result <- calculate_icc(marker_values, markerless_values)
  
  # RMSE와 ICC 값을 반환
  return(list(
    rmse_values = rmse,
    icc_values = icc_result$value,
    icc_pvalues = icc_result$p.value  # p-value 이름 수정
  ))
}

# Function to analyze joints
analyze_joints <- function(data) {
  results <- list()
  
  for (joint_name in names(data)) {
    joint_data <- data[[joint_name]]
    results[[joint_name]] <- list()
    
    # 각 coordinate(X, Y, Z)에 대해 처리
    for (coord in c("X", "Y", "Z")) {
      # cat(sprintf("\nProcessing %s - %s", joint_name, coord))
      
      if (is.null(joint_data$marker_based) || is.null(joint_data$markerless) ||
          nrow(joint_data$marker_based) == 0 || nrow(joint_data$markerless) == 0) {
        cat("\nSkipping due to missing data")
        next
      }
      
      # RMSE와 ICC 계산
      coord_results <- process_coordinate(
        joint_data$marker_based,
        joint_data$markerless,
        joint_name,
        coord
      )
      
      if (!is.null(coord_results)) {
        if (is.null(results[[joint_name]][[coord]])) {
          results[[joint_name]][[coord]] <- list(
            rmse_values = c(),
            icc_values = c(),
            icc_pvalues = c()  # p-value 저장을 위한 리스트 추가
          )
        }
        
        # 결과 저장
        results[[joint_name]][[coord]]$rmse_values <- c(
          results[[joint_name]][[coord]]$rmse_values,
          coord_results$rmse_values
        )
        results[[joint_name]][[coord]]$icc_values <- c(
          results[[joint_name]][[coord]]$icc_values,
          coord_results$icc_values
        )
        results[[joint_name]][[coord]]$icc_pvalues <- c(  # p-value 저장
          results[[joint_name]][[coord]]$icc_pvalues,
          coord_results$icc_pvalues  # 이름 수정
        )
      } else {
        cat("\nNo valid results for", joint_name, "-", coord)
      }
    }
  }
  
  return(results)
}

# Function to create RMSE plot
create_rmse_plot <- function(rmse_summary) {
  # Reshape data for plotting
  rmse_long <- rmse_summary %>%
    tidyr::pivot_longer(
      cols = starts_with("RMSE"),
      names_to = "Metric",
      values_to = "RMSE"
    ) %>%
    mutate(
      Metric = gsub("RMSE_", "", Metric),
      Joint = factor(Joint, levels = unique(Joint))
    )
  
  # Create plot
  p <- ggplot(rmse_long, aes(x = Joint, y = RMSE, fill = Metric)) +
    geom_bar(stat = "identity", position = "dodge") +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(hjust = 0.5),
      legend.position = "top"
    ) +
    labs(
      title = "RMSE by Joint and Coordinate",
      x = "Joint",
      y = "RMSE (mm)",
      fill = "Coordinate"
    )
  
  # Save plot
  ggsave("rmse_plot.png", p, width = 12, height = 8)
  
  return(p)
}

# Function to create correlation plot
create_correlation_plot <- function(correlation_summary) {
  # Reshape data for plotting
  correlation_long <- correlation_summary %>%
    tidyr::pivot_longer(
      cols = starts_with("Correlation"),
      names_to = "Metric",
      values_to = "Correlation"
    ) %>%
    mutate(
      Metric = gsub("Correlation_", "", Metric),
      Joint = factor(Joint, levels = unique(Joint))
    )
  
  # Create plot
  p <- ggplot(correlation_long, aes(x = Joint, y = Correlation, fill = Metric)) +
    geom_bar(stat = "identity", position = "dodge") +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(hjust = 0.5),
      legend.position = "top"
    ) +
    labs(
      title = "Correlation by Joint and Coordinate",
      x = "Joint",
      y = "Correlation Coefficient",
      fill = "Coordinate"
    ) +
    ylim(0, 1)  # Correlation coefficients range from 0 to 1
  
  # Save plot
  ggsave("correlation_plot.png", p, width = 12, height = 8)
  
  return(p)
}

# Function to create trajectory plots
create_trajectory_plots <- function(data, joint_name) {
  joint_data <- data[[joint_name]]
  
  # Create separate plots for each coordinate
  plots <- list()
  
  for (coord in c("X", "Y", "Z")) {
    p <- ggplot() +
      geom_line(data = joint_data$marker_based, 
                aes(x = Frame, y = .data[[coord]], color = "Marker-based"),
                linewidth = 1) +
      geom_line(data = joint_data$markerless,
                aes(x = Frame, y = .data[[coord]], color = "Markerless"),
                linewidth = 1) +
      theme_minimal() +
      theme(
        plot.title = element_text(hjust = 0.5),
        legend.position = "top"
      ) +
      labs(
        title = sprintf("%s - %s Coordinate Trajectory", joint_name, coord),
        x = "Frame",
        y = sprintf("%s Position (mm)", coord),
        color = "Method"
      )
    
    # Save plot
    ggsave(sprintf("%s_%s_trajectory.png", joint_name, coord), p, width = 12, height = 6)
    
    plots[[coord]] <- p
  }
  
  return(plots)
}

# Function to create time series plots
create_time_series_plots <- function(data, joint_name) {
  joint_data <- data[[joint_name]]
  
  # Create separate plots for each coordinate
  plots <- list()
  
  for (coord in c("X", "Y", "Z")) {
    p <- ggplot() +
      geom_line(data = joint_data$marker_based, 
                aes(x = Frame, y = .data[[coord]], color = "Marker-based"),
                linewidth = 1) +
      geom_line(data = joint_data$markerless,
                aes(x = Frame, y = .data[[coord]], color = "Markerless"),
                linewidth = 1) +
      theme_minimal() +
      theme(
        plot.title = element_text(hjust = 0.5),
        legend.position = "top"
      ) +
      labs(
        title = sprintf("%s - %s Coordinate Trajectory", joint_name, coord),
        x = "Frame",
        y = sprintf("%s Position (mm)", coord),
        color = "Method"
      )
    
    # Save plot
    ggsave(sprintf("%s_%s_trajectory.png", joint_name, coord), p, width = 12, height = 6)
    
    plots[[coord]] <- p
  }
  
  return(plots)
}

# Function to create visualization plots
create_plots <- function(summary_stats, save_dir = "visualization", bland_altman_data = NULL, plot_suffix = "") {
  if (!dir.exists(save_dir)) {
    dir.create(save_dir)
  }
  
  # Bland-Altman 플롯 생성
  if (!is.null(bland_altman_data)) {
    ba_dir <- file.path(save_dir, paste0("bland_altman", plot_suffix))
    if (!dir.exists(ba_dir)) {
      dir.create(ba_dir, recursive = TRUE)
    }
    
    for (joint in names(bland_altman_data)) {
      for (coord in names(bland_altman_data[[joint]])) {
        data <- bland_altman_data[[joint]][[coord]]
        if (nrow(data) > 0) {
          ba_results <- bland_altman_analysis(data, joint, coord)
          
          # 플롯 저장
          plot_filename <- file.path(ba_dir, 
                                   sprintf("bland_altman_%s_%s%s.png", 
                                         joint, coord, plot_suffix))
          ggsave(plot_filename, ba_results$plot, width = 10, height = 8, dpi = 300)
          
          # Save statistics to Excel
          stats_df <- data.frame(
            Joint = joint,
            Coordinate = coord,
            Mean_Difference = ba_results$statistics$mean_difference,
            SD_Difference = ba_results$statistics$sd_difference,
            Upper_LOA = ba_results$statistics$upper_loa,
            Lower_LOA = ba_results$statistics$lower_loa
          )
          
          excel_filename <- file.path(ba_dir, 
                                    sprintf("bland_altman_stats%s.xlsx", plot_suffix))
          
          if (file.exists(excel_filename)) {
            existing_stats <- read.xlsx(excel_filename)
            stats_df <- rbind(existing_stats, stats_df)
          }
          
          write.xlsx(stats_df, excel_filename, rowNames = FALSE)
        }
      }
    }
  }
  
  # Prepare data for plotting
  plot_data <- data.frame(
    Joint = character(),
    Coordinate = character(),
    Value = numeric(),
    Metric = character(),
    stringsAsFactors = FALSE
  )
  
  for (joint in names(summary_stats)) {
    for (coord in names(summary_stats[[joint]])) {
      icc_values <- summary_stats[[joint]][[coord]]$icc_values
      rmse_values <- summary_stats[[joint]][[coord]]$rmse_values
      
      temp_icc <- data.frame(
        Joint = joint,
        Coordinate = coord,
        Value = icc_values,
        Metric = "ICC",
        stringsAsFactors = FALSE
      )
      
      temp_rmse <- data.frame(
        Joint = joint,
        Coordinate = coord,
        Value = rmse_values,
        Metric = "RMSE",
        stringsAsFactors = FALSE
      )
      
      plot_data <- rbind(plot_data, temp_icc, temp_rmse)
    }
  }
  
  # Define color palette for joints
  joint_colors <- c(
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
  )
  
  # ICC와 RMSE 각각에 대해 플롯 생성
  for (metric in c("ICC", "RMSE")) {
    metric_data <- plot_data %>% filter(Metric == metric)
    
    # Create faceted plot for all joints
    p <- ggplot(metric_data, aes(x = Coordinate, y = Value, fill = Joint)) +
      # Add violin plot
      geom_violin(position = position_dodge(0.8), alpha = 0.3, scale = "width", width = 0.7) +
      # Add boxplot
      geom_boxplot(position = position_dodge(0.8), width = 0.2, alpha = 0.7, outlier.shape = NA) +
      # Add jittered points
      geom_point(position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.8), 
                alpha = 0.3, size = 1) +
      # Customize labels and theme
      labs(title = sprintf("%s (deg.)%s", metric, 
                          ifelse(plot_suffix == "", "", paste0(" - ", 
                                                             ifelse(plot_suffix == "_mean", "Mean", "Peak")))),
           x = "Coordinate",
           y = metric) +
      theme_minimal() +
      theme(
        plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
        axis.title = element_text(size = 18),
        axis.text = element_text(size = 20),
        legend.position = "right",
        legend.title = element_text(size = 16),
        legend.text = element_text(size = 14)
      ) +
      scale_fill_brewer(palette = "Set2")
    
    # Add reference line based on metric
    if (metric == "ICC") {
      p <- p + 
        ylim(0, 1) +
        geom_hline(yintercept = 0.9, linetype = "dashed", color = "red", linewidth = 0.5)
    } else {
      p <- p + 
        ylim(0, 30) +
        geom_hline(yintercept = 5, linetype = "dashed", color = "red", linewidth = 0.5)
    }
    
    # Save plot
    plot_filename <- file.path(save_dir, sprintf("%s_combined_plot%s.png", metric, plot_suffix))
    ggsave(plot_filename, p, width = 12, height = 8, dpi = 300, bg = "white")
  }
  
  return(NULL)
}

# Function to save Bland-Altman results to Excel
save_bland_altman_results_to_excel <- function(bland_altman_results, motion_type) {
  # Create results directory if it doesn't exist
  if (!dir.exists("results")) {
    dir.create("results")
  }
  
  # Initialize data frame to store results
  results_df <- data.frame(
    Joint = character(),
    Coordinate = character(),
    Mean_Difference = numeric(),
    SD_Difference = numeric(),
    Upper_LOA = numeric(),
    Lower_LOA = numeric(),
    stringsAsFactors = FALSE
  )
  
  # Collect results for each joint and coordinate
  for (joint in names(bland_altman_results)) {
    for (coord in names(bland_altman_results[[joint]])) {
      stats <- bland_altman_results[[joint]][[coord]]$statistics
      
      results_df <- rbind(results_df, data.frame(
        Joint = joint,
        Coordinate = coord,
        Mean_Difference = stats$mean_difference,
        SD_Difference = stats$sd_difference,
        Upper_LOA = stats$upper_loa,
        Lower_LOA = stats$lower_loa,
        stringsAsFactors = FALSE
      ))
    }
  }
  
  # Save to Excel
  filename <- sprintf("results/%s_bland_altman_results.xlsx", motion_type)
  write.xlsx(results_df, filename, rowNames = FALSE)
  
  cat(sprintf("\nBland-Altman results saved to: %s", filename))
}

# Save results to Excel
save_results_to_excel <- function(summary_stats, file_path = "results/analysis_results.xlsx") {
  # Create directory if it doesn't exist
  results_dir <- dirname(file_path)
  if (!dir.exists(results_dir)) {
    dir.create(results_dir, recursive = TRUE)
  }
  
  # Create workbook
  wb <- createWorkbook()
  
  # Add summary sheet
  addWorksheet(wb, "Summary")
  
  # Prepare summary data
  summary_data <- data.frame(
    Joint = character(),
    Coordinate = character(),
    RMSE_Mean = numeric(),
    RMSE_SD = numeric(),
    ICC_Mean = numeric(),
    ICC_SD = numeric(),
    ICC_P_Value = numeric(),  # p-value 열 추가
    stringsAsFactors = FALSE
  )
  
  for (joint in names(summary_stats)) {
    for (coord in names(summary_stats[[joint]])) {
      stats <- summary_stats[[joint]][[coord]]
      summary_data <- rbind(summary_data, data.frame(
        Joint = joint,
        Coordinate = coord,
        RMSE_Mean = stats$rmse_mean,
        RMSE_SD = stats$rmse_sd,
        ICC_Mean = stats$icc_mean,
        ICC_SD = stats$icc_sd,
        ICC_P_Value = stats$icc_pvalue_mean,  # p-value 저장
        stringsAsFactors = FALSE
      ))
    }
  }
  
  # Write summary data
  writeData(wb, "Summary", summary_data)
  
  # Add styling
  style_header <- createStyle(
    textDecoration = "bold",
    halign = "center",
    fgFill = "#4F81BD",
    fontColour = "white"
  )
  
  style_body <- createStyle(
    halign = "center",
    border = "TopBottom",
    borderColour = "#4F81BD"
  )
  
  # Apply styles
  addStyle(wb, "Summary", style_header, rows = 1, cols = 1:ncol(summary_data))
  addStyle(wb, "Summary", style_body, rows = 2:(nrow(summary_data) + 1), 
           cols = 1:ncol(summary_data), gridExpand = TRUE)
  
  # Auto-adjust column widths
  setColWidths(wb, "Summary", cols = 1:ncol(summary_data), widths = "auto")
  
  # Save workbook
  saveWorkbook(wb, file_path, overwrite = TRUE)
} 

# Update main function to use both peak and mean values for Bland-Altman analysis
main <- function(motion_type = NULL, subject_type = NULL) {
    tryCatch({
        cat(sprintf("\nStarting analysis for motion type: %s and subject: %s\n", 
                    motion_type, subject_type))
        
        # 1. 파일 찾기
        files_df <- find_edited_files(motion_type, subject_type)
        
        if (nrow(files_df) == 0) {
            stop("No files found matching the criteria")
        }
        
        # 파일 처리 및 데이터 로딩
        results_list <- list()
        time_series_data <- list()  # 시계열 데이터를 저장할 새로운 리스트
        bland_altman_data_peak <- list() # Peak 값의 Bland-Altman 데이터를 저장할 리스트
        bland_altman_data_mean <- list() # Mean 값의 Bland-Altman 데이터를 저장할 리스트
        
        for (i in 1:nrow(files_df)) {
            file_path <- files_df$file_path[i]
            data <- read_and_process_file(file_path)
            
            if (!is.null(data)) {
                
                # 시계열 데이터 저장
                for (joint in names(data)) {
                    if (!(joint %in% names(time_series_data))) {
                        time_series_data[[joint]] <- list()
                        bland_altman_data_peak[[joint]] <- list()
                        bland_altman_data_mean[[joint]] <- list()
                    }
                    
                    # 각 좌표(X, Y, Z)에 대해 처리
                    for (coord in c("X", "Y", "Z")) {
                        if (!(coord %in% names(bland_altman_data_peak[[joint]]))) {
                            bland_altman_data_peak[[joint]][[coord]] <- data.frame(
                                marker_value = numeric(),
                                markerless_value = numeric()
                            )
                            bland_altman_data_mean[[joint]][[coord]] <- data.frame(
                                marker_value = numeric(),
                                markerless_value = numeric()
                            )
                        }
                        
                        # Peak 값 저장
                        marker_peak <- data[[joint]]$marker_peaks[[coord]]
                        markerless_peak <- data[[joint]]$markerless_peaks[[coord]]
                        
                        # Mean 값 계산 및 저장
                        marker_mean <- find_mean_value(data[[joint]]$marker_based[[coord]])
                        markerless_mean <- find_mean_value(data[[joint]]$markerless[[coord]])
                        
                        if (!is.null(marker_peak) && !is.null(markerless_peak)) {
                            bland_altman_data_peak[[joint]][[coord]] <- rbind(
                                bland_altman_data_peak[[joint]][[coord]],
                                data.frame(
                                    marker_value = marker_peak,
                                    markerless_value = markerless_peak
                                )
                            )
                        }
                        
                        if (!is.null(marker_mean) && !is.null(markerless_mean)) {
                            bland_altman_data_mean[[joint]][[coord]] <- rbind(
                                bland_altman_data_mean[[joint]][[coord]],
                                data.frame(
                                    marker_value = marker_mean,
                                    markerless_value = markerless_mean
                                )
                            )
                        }
                    }
                    
                    # 원본 시계열 데이터 저장
                    time_series_data[[joint]][[basename(file_path)]] <- list(
                        marker_based = data[[joint]]$marker_based,
                        markerless = data[[joint]]$markerless
                    )
                }
                
                # 기존 분석 결과 저장
                file_results <- analyze_joints(data)
                if (!is.null(file_results)) {
                    results_list[[basename(file_path)]] <- file_results
                }
            }
        }
        
        # 4. 모든 결과를 통합하여 ICC 및 RMSE 계산
        combined_results <- list()
        
        for (file_name in names(results_list)) {
            file_results <- results_list[[file_name]]
            
            for (joint in names(file_results)) {
                if (is.null(combined_results[[joint]])) {
                    combined_results[[joint]] <- list()
                }
                
                for (coord in names(file_results[[joint]])) {
                    coord_results <- file_results[[joint]][[coord]]
                    
                    if (is.null(combined_results[[joint]][[coord]])) {
                        combined_results[[joint]][[coord]] <- list(
                            icc_values = coord_results$icc_values,
                            icc_pvalues = coord_results$icc_pvalues,
                            rmse_values = coord_results$rmse_values
                        )
                    } else {
                        combined_results[[joint]][[coord]]$icc_values <- c(
                            combined_results[[joint]][[coord]]$icc_values,
                            coord_results$icc_values
                        )
                        combined_results[[joint]][[coord]]$icc_pvalues <- c(
                            combined_results[[joint]][[coord]]$icc_pvalues,
                            coord_results$icc_pvalues
                        )
                        combined_results[[joint]][[coord]]$rmse_values <- c(
                            combined_results[[joint]][[coord]]$rmse_values,
                            coord_results$rmse_values
                        )
                    }
                }
            }
        }
        
        # 5. 최종 결과 계산 (평균 및 표준편차)
        summary_stats <- list()
        
        for (joint in names(combined_results)) {
            summary_stats[[joint]] <- list()
            
            for (coord in names(combined_results[[joint]])) {
                icc_values <- combined_results[[joint]][[coord]]$icc_values
                icc_pvalues <- combined_results[[joint]][[coord]]$icc_pvalues
                rmse_values <- combined_results[[joint]][[coord]]$rmse_values
                
                summary_stats[[joint]][[coord]] <- list(
                    icc_values = icc_values,
                    icc_pvalues = icc_pvalues,
                    rmse_values = rmse_values,
                    icc_mean = mean(icc_values, na.rm = TRUE),
                    icc_sd = sd(icc_values, na.rm = TRUE),
                    icc_pvalue_mean = mean(icc_pvalues, na.rm = TRUE),
                    rmse_mean = mean(rmse_values, na.rm = TRUE),
                    rmse_sd = sd(rmse_values, na.rm = TRUE)
                )
            }
        }
        
        # 6. 결과 출력
        cat("\n=== Analysis Results ===\n")
        for (joint in names(summary_stats)) {
            cat(sprintf("\n%s:\n", joint))
            for (coord in names(summary_stats[[joint]])) {
                stats <- summary_stats[[joint]][[coord]]
                cat(sprintf("  %s coordinate:\n", coord))
                cat(sprintf("    RMSE: %.2f ± %.2f mm\n", 
                       stats$rmse_mean, stats$rmse_sd))
                cat(sprintf("    ICC: %.3f ± %.3f (p = %.3f)\n", 
                       stats$icc_mean, stats$icc_sd, stats$icc_pvalue_mean))
            }
        }
        
        # 7. 시각화 및 결과 저장
        create_plots(summary_stats, bland_altman_data = bland_altman_data_peak)
        create_plots(summary_stats, bland_altman_data = bland_altman_data_mean, plot_suffix = "_mean")
        save_results_to_excel(summary_stats)
        
        cat("\nAnalysis complete. Results have been saved.\n")
        
        return(list(
            summary_stats = summary_stats,
            time_series_data = time_series_data,
            bland_altman_data_peak = bland_altman_data_peak,
            bland_altman_data_mean = bland_altman_data_mean
        ))
        
    }, error = function(e) {
        cat("\nError in main function:", e$message, "\n")
        return(NULL)
    })
}

# Function to remove outliers using IQR method
remove_outliers <- function(data) {
  # Calculate differences
  differences <- data$marker_value - data$markerless_value
  
  # Calculate Q1, Q3, and IQR for differences
  Q1 <- quantile(differences, 0.25)
  Q3 <- quantile(differences, 0.75)
  IQR <- Q3 - Q1
  
  # Define bounds
  lower_bound <- Q1 - (2.0 * IQR)
  upper_bound <- Q3 + (2.0 * IQR)
  
  # Keep only the data within bounds
  valid_indices <- differences >= lower_bound & differences <= upper_bound
  
  # Return filtered data
  return(data[valid_indices, ])
}

# Function to analyze all subjects for a given motion type
analyze_all_subjects <- function(motion_type = "kicking") {
  # Get list of all subjects
  subjects <- c("고승준", "김리언", "김준성", "김태형", 
                "방민식", "성기훈", "이준석", "장귀현",
                "김건희", "김훈민")
  
  # Initialize empty list to store all results
  all_results <- list()
  combined_summary_stats <- list()
  
  # Initialize combined Bland-Altman data for both peak and mean
  combined_bland_altman_data_peak <- list()
  combined_bland_altman_data_mean <- list()
  
  # Process each subject
  for (subject in subjects) {
    cat(sprintf("\n\nProcessing subject: %s\n", subject))
    
    # Run analysis for current subject
    tryCatch({
      result <- main(motion_type, subject)
      if (!is.null(result)) {
        all_results[[subject]] <- result
        
        # Combine Bland-Altman data for peak values
        for (joint in names(result$bland_altman_data_peak)) {
          if (!(joint %in% names(combined_bland_altman_data_peak))) {
            combined_bland_altman_data_peak[[joint]] <- list()
          }
          
          for (coord in names(result$bland_altman_data_peak[[joint]])) {
            if (!(coord %in% names(combined_bland_altman_data_peak[[joint]]))) {
              combined_bland_altman_data_peak[[joint]][[coord]] <- data.frame(
                marker_value = numeric(),
                markerless_value = numeric(),
                subject = character()
              )
            }
            
            data <- result$bland_altman_data_peak[[joint]][[coord]]
            if (nrow(data) > 0) {
              data$subject <- subject
              combined_bland_altman_data_peak[[joint]][[coord]] <- rbind(
                combined_bland_altman_data_peak[[joint]][[coord]],
                data
              )
            }
          }
        }
        
        # Combine Bland-Altman data for mean values
        for (joint in names(result$bland_altman_data_mean)) {
          if (!(joint %in% names(combined_bland_altman_data_mean))) {
            combined_bland_altman_data_mean[[joint]] <- list()
          }
          
          for (coord in names(result$bland_altman_data_mean[[joint]])) {
            if (!(coord %in% names(combined_bland_altman_data_mean[[joint]]))) {
              combined_bland_altman_data_mean[[joint]][[coord]] <- data.frame(
                marker_value = numeric(),
                markerless_value = numeric(),
                subject = character()
              )
            }
            
            data <- result$bland_altman_data_mean[[joint]][[coord]]
            if (nrow(data) > 0) {
              data$subject <- subject
              combined_bland_altman_data_mean[[joint]][[coord]] <- rbind(
                combined_bland_altman_data_mean[[joint]][[coord]],
                data
              )
            }
          }
        }
        
        # Combine results
        for (joint in names(result$summary_stats)) {
          if (!(joint %in% names(combined_summary_stats))) {
            combined_summary_stats[[joint]] <- list()
          }
          
          for (coord in names(result$summary_stats[[joint]])) {
            if (!(coord %in% names(combined_summary_stats[[joint]]))) {
              combined_summary_stats[[joint]][[coord]] <- list(
                icc_values = c(),
                icc_pvalues = c(),
                rmse_values = c()
              )
            }
            
            # Append values from this subject
            combined_summary_stats[[joint]][[coord]]$icc_values <- c(
              combined_summary_stats[[joint]][[coord]]$icc_values,
              result$summary_stats[[joint]][[coord]]$icc_values
            )
            combined_summary_stats[[joint]][[coord]]$icc_pvalues <- c(
              combined_summary_stats[[joint]][[coord]]$icc_pvalues,
              result$summary_stats[[joint]][[coord]]$icc_pvalues
            )
            combined_summary_stats[[joint]][[coord]]$rmse_values <- c(
              combined_summary_stats[[joint]][[coord]]$rmse_values,
              result$summary_stats[[joint]][[coord]]$rmse_values
            )
          }
        }
      }
    }, error = function(e) {
      cat(sprintf("Error processing subject %s: %s\n", subject, e$message))
    })
  }
  
  # Calculate overall means and SDs for all subjects combined
  for (joint in names(combined_summary_stats)) {
    for (coord in names(combined_summary_stats[[joint]])) {
      icc_values <- combined_summary_stats[[joint]][[coord]]$icc_values
      icc_pvalues <- combined_summary_stats[[joint]][[coord]]$icc_pvalues
      rmse_values <- combined_summary_stats[[joint]][[coord]]$rmse_values
      
      combined_summary_stats[[joint]][[coord]]$icc_mean <- mean(icc_values, na.rm = TRUE)
      combined_summary_stats[[joint]][[coord]]$icc_sd <- sd(icc_values, na.rm = TRUE)
      combined_summary_stats[[joint]][[coord]]$icc_pvalue_mean <- mean(icc_pvalues, na.rm = TRUE)
      combined_summary_stats[[joint]][[coord]]$rmse_mean <- mean(rmse_values, na.rm = TRUE)
      combined_summary_stats[[joint]][[coord]]$rmse_sd <- sd(rmse_values, na.rm = TRUE)
    }
  }
  
  # Create visualization directory for all subjects
  vis_dir <- sprintf("visualization_all_subjects_%s", motion_type)
  if (!dir.exists(vis_dir)) {
    dir.create(vis_dir)
  }
  
  # Perform Bland-Altman analysis on combined data for peak values
  combined_bland_altman_results_peak <- list()
  for (joint in names(combined_bland_altman_data_peak)) {
    combined_bland_altman_results_peak[[joint]] <- list()
    for (coord in names(combined_bland_altman_data_peak[[joint]])) {
      data <- combined_bland_altman_data_peak[[joint]][[coord]]
      if (nrow(data) > 0) {
        # Remove outliers
        data <- remove_outliers(data)
        
        ba_results <- bland_altman_analysis(data, joint, coord)
        combined_bland_altman_results_peak[[joint]][[coord]] <- ba_results
      }
    }
  }
  
  # Perform Bland-Altman analysis on combined data for mean values
  combined_bland_altman_results_mean <- list()
  for (joint in names(combined_bland_altman_data_mean)) {
    combined_bland_altman_results_mean[[joint]] <- list()
    for (coord in names(combined_bland_altman_data_mean[[joint]])) {
      data <- combined_bland_altman_data_mean[[joint]][[coord]]
      if (nrow(data) > 0) {
        # Remove outliers
        data <- remove_outliers(data)
        
        ba_results <- bland_altman_analysis(data, joint, coord)
        combined_bland_altman_results_mean[[joint]][[coord]] <- ba_results
      }
    }
  }
  
  # Save combined results to Excel
  save_results_to_excel(combined_summary_stats, 
                       file_path = sprintf("results/%s_all_subjects_results.xlsx", motion_type))
  
  # Save Bland-Altman results to Excel for both peak and mean
  save_bland_altman_results_to_excel(combined_bland_altman_results_peak, 
                                   paste0(motion_type, "_peak"))
  save_bland_altman_results_to_excel(combined_bland_altman_results_mean, 
                                   paste0(motion_type, "_mean"))
  
  # Create plots with the combined data
  create_plots(combined_summary_stats, 
              bland_altman_data = combined_bland_altman_data_peak,
              save_dir = vis_dir)
  create_plots(combined_summary_stats, 
              bland_altman_data = combined_bland_altman_data_mean,
              save_dir = vis_dir,
              plot_suffix = "_mean")
  
  # Create time series plots with mean and SD
  create_time_series_with_stats(all_results, motion_type)
  
  # Print overall results
  cat("\n=== Overall Analysis Results (All Subjects) ===\n")
  for (joint in names(combined_summary_stats)) {
    cat(sprintf("\n%s:\n", joint))
    for (coord in names(combined_summary_stats[[joint]])) {
      stats <- combined_summary_stats[[joint]][[coord]]
      cat(sprintf("  %s coordinate:\n", coord))
      cat(sprintf("    RMSE: %.2f ± %.2f mm\n", 
                stats$rmse_mean, stats$rmse_sd))
      cat(sprintf("    ICC: %.3f ± %.3f (p = %.3f)\n", 
                stats$icc_mean, stats$icc_sd, stats$icc_pvalue_mean))
    }
  }
  
  return(all_results)
}

create_combined_plots <- function(plot_data, save_dir = "visualization_all_subjects") {
  if (!dir.exists(save_dir)) {
    dir.create(save_dir)
  }
  
  # Joint 순서 정의
  joint_order <- c("Ankle", "Knee", "Hip", "Trunk")
  # Coordinate 순서 정의
  coord_order <- c("X", "Y", "Z")
  
  # 데이터 전처리
  plot_data <- plot_data %>%
    mutate(
      Joint = factor(Joint, levels = joint_order),
      Coordinate = factor(Coordinate, levels = coord_order),
      x_label = interaction(Joint, Coordinate, sep = "_")
    )
  
  # ICC와 RMSE 데이터 분리
  icc_data <- plot_data %>% filter(Metric == "ICC")
  rmse_data <- plot_data %>% filter(Metric == "RMSE")
  
  # RMSE의 최대값을 30으로 고정
  rmse_max <- 30
  
  # RMSE 데이터를 ICC 스케일로 변환 (0-1 스케일)
  rmse_scaled <- rmse_data %>%
    mutate(Value_scaled = Value / rmse_max)
  
  # Create combined plot
    p <- ggplot() +
    # ICC 관련 레이어
    geom_violin(data = icc_data, aes(x = Coordinate, y = Value, fill = Joint),
               position = position_dodge(0.8), alpha = 0.3, scale = "width", width = 0.7) +
    geom_boxplot(data = icc_data, aes(x = Coordinate, y = Value, fill = Joint),
                position = position_dodge(0.8), width = 0.2, alpha = 0.7, outlier.shape = NA) +
    stat_summary(data = icc_data, aes(x = Coordinate, y = Value, group = Joint),
                fun = mean, geom = "point", shape = 23, size = 3,
                position = position_dodge(0.8), fill = "white", color = "black") +
    geom_point(data = icc_data, aes(x = Coordinate, y = Value, fill = Joint),
              position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.8),
              alpha = 0.3, size = 1) +
    # RMSE 관련 레이어
    geom_col(data = rmse_scaled %>% group_by(Joint, Coordinate) %>% 
               summarise(Value_scaled = mean(Value_scaled), .groups = 'drop'),
             aes(x = Coordinate, y = Value_scaled, fill = Joint),
             position = position_dodge(0.8), alpha = 0.3, width = 0.6) +
    # 두 개의 y축 스케일 설정
    scale_y_continuous(name = "ICC",
                      limits = c(0, 1),
                      sec.axis = sec_axis(~ . * rmse_max, name = "RMSE (deg.)",
                                        breaks = seq(0, rmse_max, by = 5))) +
    # Reference lines
    geom_hline(yintercept = 0.9, linetype = "dashed", color = "red", linewidth = 0.5) +
    geom_hline(yintercept = 5/rmse_max, linetype = "dashed", color = "red", linewidth = 0.5) +
    # Facet and theme
    facet_grid(. ~ Joint, scales = "free_x", space = "free_x") +
      theme_minimal() +
      theme(
      plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
      axis.title = element_text(size = 18),
      axis.text = element_text(size = 16),
      axis.text.x = element_text(angle = 0, hjust = 0.5),
      axis.title.y.right = element_text(size = 16),
      axis.text.y.right = element_text(size = 16),
      legend.position = "none",
      panel.spacing = unit(1, "lines"),
      strip.text = element_text(size = 16, face = "bold"),
      plot.caption = element_text(size = 14, hjust = 1),
      panel.grid.major.x = element_blank()
    ) +
    scale_fill_manual(values = c(
      "Ankle" = "#1f77b4",
      "Knee" = "#2ca02c",
      "Hip" = "#ff7f0e",
      "Trunk" = "#d62728"
    )) +
    labs(title = "ICC and RMSE (deg.) - All Subjects",
         x = "Joint",
         caption = "Box: 25-75th percentile, Line: median, Diamond: mean\nBars: RMSE mean values")
    
    # Save plot
  plot_filename <- file.path(save_dir, "ICC_RMSE_combined_plot.png")
  ggsave(plot_filename, p, width = 15, height = 8, dpi = 300, bg = "white")
}

create_time_series_with_stats <- function(all_results, motion_type) {
  if (!dir.exists("time_series_stats")) {
    dir.create("time_series_stats")
  }
  
  # Create Excel directory if it doesn't exist
  excel_dir <- "time_series_stats/excel"
  if (!dir.exists(excel_dir)) {
    dir.create(excel_dir, recursive = TRUE)
  }
   
  # Initialize combined data structure
  combined_data <- list()
  
  # Collect all normalized data (already 101 frames)
  for (subject in names(all_results)) {
    if (!is.null(all_results[[subject]]$time_series_data)) {
      for (joint in names(all_results[[subject]]$time_series_data)) {
      if (!(joint %in% names(combined_data))) {
        combined_data[[joint]] <- list()
          for (coord in c("X", "Y", "Z")) {
          combined_data[[joint]][[coord]] <- list(
              marker = matrix(nrow = 101, ncol = 0),
              markerless = matrix(nrow = 101, ncol = 0)
            )
          }
        }
        
        # Get all trials for this subject
        trials <- all_results[[subject]]$time_series_data[[joint]]
        for (trial_data in trials) {
          for (coord in c("X", "Y", "Z")) {
            if (!is.null(trial_data$marker_based[[coord]]) && 
                !is.null(trial_data$markerless[[coord]])) {
              # Add trial data to matrices
              combined_data[[joint]][[coord]]$marker <- cbind(
                combined_data[[joint]][[coord]]$marker,
                trial_data$marker_based[[coord]]
              )
              combined_data[[joint]][[coord]]$markerless <- cbind(
                combined_data[[joint]][[coord]]$markerless,
                trial_data$markerless[[coord]]
              )
            }
          }
        }
      }
    }
  }
  
  # Create plots for each joint and coordinate
  for (joint in names(combined_data)) {
    for (coord in c("X", "Y", "Z")) {
      cat(sprintf("\nProcessing %s - %s:", joint, coord))
      
      marker_data <- combined_data[[joint]][[coord]]$marker
      markerless_data <- combined_data[[joint]][[coord]]$markerless
      
      n_trials <- ncol(marker_data)
      cat(sprintf("\nNumber of trials: %d", n_trials))
      
      # Calculate mean and SD for each frame
      marker_mean <- rowMeans(marker_data, na.rm = TRUE)
      markerless_mean <- rowMeans(markerless_data, na.rm = TRUE)
      marker_sd <- apply(marker_data, 1, sd, na.rm = TRUE)
      markerless_sd <- apply(markerless_data, 1, sd, na.rm = TRUE)
      
      # Create plot data
      plot_data <- data.frame(
        Frame = 0:100,  # Convert to percentage
        Marker = marker_mean,
        Markerless = markerless_mean,
        Marker_SD = marker_sd,
        Markerless_SD = markerless_sd
      )
      
      # Create plot
      p <- ggplot(plot_data, aes(x = Frame)) +
        # Add SD ribbons
        geom_ribbon(
          aes(ymin = Marker - Marker_SD,
              ymax = Marker + Marker_SD,
              fill = "Marker-based"),
          alpha = 0.2
        ) +
        geom_ribbon(
          aes(ymin = Markerless - Markerless_SD,
              ymax = Markerless + Markerless_SD,
              fill = "Markerless"),
          alpha = 0.2
        ) +
        # Add mean lines
        geom_line(aes(y = Marker, color = "Marker-based"), linewidth = 1) +
        geom_line(aes(y = Markerless, color = "Markerless"), linewidth = 1) +
        
        # Define colors
        scale_color_manual(
          name = "Measurement Method",
          values = c("Marker-based" = "blue", "Markerless" = "red")
        ) +
        scale_fill_manual(
          name = "Measurement Method",
          values = c("Marker-based" = "blue", "Markerless" = "red")
        ) +
        
        # Combine legends
        guides(
          color = guide_legend(override.aes = list(fill = NA)),
          fill = guide_legend(override.aes = list(alpha = 0.5))
        ) +
        
        # Labels and theme
        labs(
          title = sprintf("%s - %s (%s)", joint, coord, motion_type),
          subtitle = sprintf("Mean ± SD across all subjects (n=%d)", n_trials),
          x = "Normalized Frame (%)",
          y = "Angle (degrees)"
        ) +
        theme_minimal() +
        theme(
          text = element_text(size = 16),
          plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
          plot.subtitle = element_text(size = 16, hjust = 0.5),
          legend.position = "right",
          legend.box = "vertical",
          panel.grid.major = element_line(color = "grey90"),
          panel.grid.minor = element_line(color = "grey95"),
          panel.background = element_rect(fill = "white"),
          plot.background = element_rect(fill = "white")
        ) +
        scale_x_continuous(breaks = seq(0, 100, 20))  # Add breaks every 20%
    
    # Save plot
      filename <- sprintf("time_series_stats/%s_%s_%s_time_series.png", 
                        motion_type, joint, coord)
      ggsave(filename, p, width = 12, height = 8, dpi = 300, bg = "white")
      
      # Save data to Excel
      excel_data <- data.frame(
        Frame = 0:100,
        Marker_Mean = marker_mean,
        Marker_SD = marker_sd,
        Markerless_Mean = markerless_mean,
        Markerless_SD = markerless_sd
      )
      
      excel_filename <- file.path(excel_dir, 
                                sprintf("%s_%s_%s_time_series.xlsx", 
                                      motion_type, joint, coord))
      write.xlsx(excel_data, excel_filename, rowNames = FALSE)
    }
  }
}