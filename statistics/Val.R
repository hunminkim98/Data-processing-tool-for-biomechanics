# ----------------------------------------------------
# 필요한 패키지 설치/로드
# ----------------------------------------------------
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("lme4")) install.packages("lme4")
if (!require("BlandAltmanLeh")) install.packages("BlandAltmanLeh")
if (!require("irr")) install.packages("irr")
if (!require("readxl")) install.packages("readxl")
if (!require("openxlsx")) install.packages("openxlsx")
if (!require("zoo")) install.packages("zoo")

library(tidyverse)
library(ggplot2)
library(lme4)
library(BlandAltmanLeh)
library(irr)
library(readxl)
library(openxlsx)
library(zoo)

# ----------------------------------------------------
# 전역 디렉토리 설정 (필요 시 수정)
# ----------------------------------------------------
parent_dir <- "D:/석사/석사3차/Markerless validation/Results/Final2/merged_check"

# ----------------------------------------------------
# 함수 1) _edited.xlsx 파일 찾기
# ----------------------------------------------------
find_edited_files <- function(motion_type = NULL, subject_type = NULL) {
  base_dir <- parent_dir
  
  motion_pattern <- if (!is.null(motion_type)) {
    paste0(".*", motion_type, ".*")
  } else {
    ".*"
  }
  
  subject_pattern <- if (!is.null(subject_type)) {
    paste0(".*", subject_type, ".*")
  } else {
    ".*"
  }
  
  files <- list.files(
    path = base_dir,
    pattern = "_edited\\.xlsx$",
    recursive = TRUE,
    full.names = TRUE
  )
  
  filtered_files <- files[
    grepl(motion_pattern, basename(files), ignore.case = TRUE) &
    grepl(subject_pattern, dirname(files), ignore.case = TRUE)
  ]
  
  # "plots" 경로 제외
  filtered_files <- filtered_files[!grepl("plots", filtered_files, ignore.case = TRUE)]
  
  cat("\nFound files:\n")
  for (file in filtered_files) {
    cat("  ", basename(file), "\n")
  }
  
  data.frame(file_path = filtered_files, stringsAsFactors = FALSE)
}

# ----------------------------------------------------
# 함수 2) 3D 좌표(벡터 크기) 계산 헬퍼
# ----------------------------------------------------
compute_3d_magnitude <- function(df) {
  sqrt(df$X^2 + df$Y^2 + df$Z^2)
}

# ----------------------------------------------------
# 함수 3) Excel 파일의 각 시트 읽고, L/R 평균 -> 조인트 통합
#         + 3D 컬럼 추가
# ----------------------------------------------------
read_and_process_file <- function(file_path) {
  if (!file.exists(file_path)) {
    warning("File does not exist: ", file_path)
    return(NULL)
  }
  
  tryCatch({
    sheets <- excel_sheets(file_path)
    data <- list()
    
    # 필요 시 특정 subject, motion 별로 제외할 시트 처리
    is_subject_kiHoon <- grepl("성기훈", file_path, fixed = TRUE)
    is_subject_Ryan   <- grepl("김리언", file_path, fixed = TRUE)
    is_subject_TH     <- grepl("김태형", file_path, fixed = TRUE)
    is_swing_motion   <- grepl("swing", file_path, ignore.case = TRUE)
    
    processed_joints <- list()
    
    for (sheet in sheets) {
      # (1) shoulder 시트 제외
      if (grepl("shoulder", sheet, ignore.case = TRUE)) {
        next
      }
      # (2) 성기훈(swing) -> Left Knee 제외
      if (is_subject_kiHoon && is_swing_motion && grepl("knee", sheet, ignore.case = TRUE)) {
        if (!grepl("Right", sheet, ignore.case = TRUE)) {
          next
        }
      }
      # (3) 김리언(swing) -> ankle 제외
      if (is_subject_Ryan && is_swing_motion && grepl("ankle", sheet, ignore.case = TRUE)) {
        next
      }
      # (4) 김태형(swing) -> trunk 제외
      if (is_subject_TH && is_swing_motion && grepl("trunk", sheet, ignore.case = TRUE)) {
        next
      }
      
      sheet_data <- read_excel(file_path, sheet = sheet, col_names = FALSE, .name_repair = "minimal")
      data_rows  <- sheet_data[-(1:3), ]
      
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
      marker_data    <- marker_data[valid_rows, ]
      markerless_data<- markerless_data[valid_rows, ]
      
      base_joint_name <- gsub("^(Left|Right)_", "", sheet)
      
      # 아직 없는 joint면 초기화
      if (!(base_joint_name %in% names(processed_joints))) {
        processed_joints[[base_joint_name]] <- list(
          marker_based = marker_data,
          markerless   = markerless_data
        )
      } else {
        # 이미 있으면, L/R 프레임 겹치는 구간 찾아 평균
        existing <- processed_joints[[base_joint_name]]
        common_frames <- intersect(marker_data$Frame, existing$marker_based$Frame)
        
        if (length(common_frames) > 0) {
          marker_subset    <- marker_data[marker_data$Frame %in% common_frames, ]
          markerless_subset<- markerless_data[markerless_data$Frame %in% common_frames, ]
          existing_marker  <- existing$marker_based[existing$marker_based$Frame %in% common_frames, ]
          existing_markerless <- existing$markerless[existing$markerless$Frame %in% common_frames, ]
          
          processed_joints[[base_joint_name]] <- list(
            marker_based = data.frame(
              Frame = common_frames,
              X = (existing_marker$X + marker_subset$X)/2,
              Y = (existing_marker$Y + marker_subset$Y)/2,
              Z = (existing_marker$Z + marker_subset$Z)/2
            ),
            markerless = data.frame(
              Frame = common_frames,
              X = (existing_markerless$X + markerless_subset$X)/2,
              Y = (existing_markerless$Y + markerless_subset$Y)/2,
              Z = (existing_markerless$Z + markerless_subset$Z)/2
            )
          )
        }
      }
    }
    
    # 각 joint에 대해 3D 컬럼 추가
    for (joint_name in names(processed_joints)) {
      mk <- processed_joints[[joint_name]]$marker_based
      ml <- processed_joints[[joint_name]]$markerless
      mk$`3D` <- compute_3d_magnitude(mk)
      ml$`3D` <- compute_3d_magnitude(ml)
      
      processed_joints[[joint_name]]$marker_based <- mk
      processed_joints[[joint_name]]$markerless   <- ml
    }
    
    # 반환 구조
    for (joint_name in names(processed_joints)) {
      data[[joint_name]] <- list(
        marker_based = processed_joints[[joint_name]]$marker_based,
        markerless   = processed_joints[[joint_name]]$markerless
      )
    }
    
    return(data)
  }, error = function(e) {
    warning("Error processing file:", file_path, "\nError:", e$message)
    return(NULL)
  })
}

# ----------------------------------------------------
# 함수 4) ICC, RMSE 계산
# ----------------------------------------------------
calculate_icc <- function(reference, comparison) {
  valid_indices <- !is.na(reference) & !is.na(comparison)
  reference  <- reference[valid_indices]
  comparison <- comparison[valid_indices]
  
  data_matrix <- data.frame(reference = reference, comparison = comparison)
  
  icc_result <- irr::icc(data_matrix,
                         model = "twoway",  
                         type  = "consistency",
                         unit  = "average")
  list(value = icc_result$value, p.value = icc_result$p.value)
}

process_coordinate <- function(marker_data, markerless_data, joint, coord) {
  marker_values    <- marker_data[[coord]]
  markerless_values<- markerless_data[[coord]]
  
  if (is.null(marker_values) || is.null(markerless_values) ||
      length(marker_values) == 0 || length(markerless_values) == 0) {
    cat("\nMissing data for", joint, "-", coord)
    return(NULL)
  }
  
  valid_indices <- !is.na(marker_values) & !is.na(markerless_values)
  marker_values    <- marker_values[valid_indices]
  markerless_values<- markerless_values[valid_indices]
  
  if (length(marker_values) < 2) {
    cat("\nNot enough data points for", joint, "-", coord)
    return(NULL)
  }
  
  rmse <- sqrt(mean((marker_values - markerless_values)^2, na.rm = TRUE))
  icc_res <- calculate_icc(marker_values, markerless_values)
  
  list(
    rmse_values = rmse,
    icc_values  = icc_res$value,
    icc_pvalues = icc_res$p.value
  )
}

analyze_joints <- function(data) {
  # data: list(joint_name -> (marker_based, markerless))
  results <- list()
  
  for (joint_name in names(data)) {
    joint_data <- data[[joint_name]]
    results[[joint_name]] <- list()
    
    for (coord in c("X","Y","Z","3D")) {
      if (nrow(joint_data$marker_based) == 0 || nrow(joint_data$markerless) == 0) {
        next
      }
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
            icc_values  = c(),
            icc_pvalues = c()
          )
        }
        results[[joint_name]][[coord]]$rmse_values <- c(
          results[[joint_name]][[coord]]$rmse_values,
          coord_results$rmse_values
        )
        results[[joint_name]][[coord]]$icc_values <- c(
          results[[joint_name]][[coord]]$icc_values,
          coord_results$icc_values
        )
        results[[joint_name]][[coord]]$icc_pvalues <- c(
          results[[joint_name]][[coord]]$icc_pvalues,
          coord_results$icc_pvalues
        )
      }
    }
  }
  
  results
}

# ----------------------------------------------------
# 함수 5) Bland-Altman 분석/플롯
# ----------------------------------------------------
bland_altman_analysis <- function(data, joint_name = "", axis_name = "") {
  differences <- data$marker_value - data$markerless_value
  means       <- (data$marker_value + data$markerless_value)/2
  
  mean_diff <- mean(differences, na.rm=TRUE)
  sd_diff   <- sd(differences, na.rm=TRUE)
  upper_loa <- mean_diff + 1.96 * sd_diff
  lower_loa <- mean_diff - 1.96 * sd_diff
  
  lines_df <- data.frame(
    yintercept = c(mean_diff, upper_loa, lower_loa),
    Line       = factor(c("Mean difference","Upper LOA","Lower LOA"),
                  levels=c("Mean difference","Upper LOA","Lower LOA"))
  )
  stats_labels <- c(
    sprintf("Mean diff: %.2f", mean_diff),
    sprintf("Upper LOA: %.2f", upper_loa),
    sprintf("Lower LOA: %.2f", lower_loa)
  )
  
  plot_title <- sprintf("Bland-Altman: %s-%s", joint_name, axis_name)
  
  ba_plot <- ggplot(data.frame(means=means, differences=differences),
                    aes(x=means, y=differences)) +
    geom_point(alpha=0.5) +
    geom_hline(data=lines_df, aes(yintercept=yintercept, color=Line, linetype=Line)) +
    geom_hline(yintercept=0, color="grey50") +
    scale_color_manual(values=c("blue","red","red"), labels=stats_labels) +
    scale_linetype_manual(values=c("solid","dashed","dashed"), labels=stats_labels) +
    labs(title=plot_title, x="Mean of Two Methods", y="Difference (Marker - Markerless)") +
    theme_minimal() +
    theme(legend.position="top")
  
  list(
    plot = ba_plot,
    statistics = list(
      mean_difference = mean_diff,
      sd_difference   = sd_diff,
      upper_loa       = upper_loa,
      lower_loa       = lower_loa
    )
  )
}

# ----------------------------------------------------
# 함수 6) 여러 그래프 생성: Bland-Altman, Combined, etc.
# ----------------------------------------------------
# (A) Combined 그래프 예시 (RMSE 통합 바 그래프)
create_combined_plot_example <- function(summary_stats, output_dir = "visualization/combined") {
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  plot_data <- data.frame(
    Joint      = character(),
    Coordinate = character(),
    RMSE       = numeric(),
    stringsAsFactors = FALSE
  )
  for (joint in names(summary_stats)) {
    for (coord in names(summary_stats[[joint]])) {
      rm <- summary_stats[[joint]][[coord]]$rmse_mean
      plot_data <- rbind(plot_data, data.frame(
        Joint=joint, Coordinate=coord, RMSE=rm
      ))
    }
  }
  
  p <- ggplot(plot_data, aes(x=paste(Joint, Coordinate, sep="_"), y=RMSE, fill=Joint)) +
    geom_bar(stat="identity", position="dodge") +
    theme_minimal() +
    labs(title="Combined RMSE Plot", x="Joint_Coord", y="RMSE (Mean)") +
    theme(axis.text.x = element_text(angle=45, hjust=1))
  
  out_file <- file.path(output_dir, "combined_rmse_plot.png")
  ggsave(out_file, p, width=8, height=5, dpi=300)
  message(">> Combined plot saved: ", out_file)
  
  return(out_file)
}

# ----------------------------------------------------
# (B) Bland-Altman 그래프 생성 후 PNG 저장
#     (bland_altman_data 구조: bland_altman_data[[joint]][[coord]] = df(marker_value, markerless_value))
# ----------------------------------------------------
create_bland_altman_plots <- function(bland_altman_data, save_dir="visualization/bland_altman") {
  if (!dir.exists(save_dir)) dir.create(save_dir, recursive=TRUE)
  
  plot_paths <- c()
  
  for (joint in names(bland_altman_data)) {
    for (coord in names(bland_altman_data[[joint]])) {
      df <- bland_altman_data[[joint]][[coord]]
      if (nrow(df) > 0) {
        ba_res <- bland_altman_analysis(df, joint, coord)
        out_file <- file.path(save_dir, sprintf("bland_altman_%s_%s.png", joint, coord))
        ggsave(out_file, ba_res$plot, width=8, height=5, dpi=300)
        
        plot_paths <- c(plot_paths, out_file)
      }
    }
  }
  plot_paths
}

# ----------------------------------------------------
# 함수 7) 최종 엑셀에 "요약통계" + "여러 그래프"를 삽입
# ----------------------------------------------------
save_all_results_to_excel <- function(summary_stats,
                                      plot_image_paths = NULL,
                                      excel_file = "results/final_results_with_plots.xlsx") {
  wb <- createWorkbook()
  
  # (1) Summary 시트
  addWorksheet(wb, "Summary")
  
  summary_data <- data.frame(
    Joint       = character(),
    Coordinate  = character(),
    RMSE_Mean   = numeric(),
    RMSE_SD     = numeric(),
    ICC_Mean    = numeric(),
    ICC_SD      = numeric(),
    ICC_P_Value = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (joint in names(summary_stats)) {
    for (coord in names(summary_stats[[joint]])) {
      st <- summary_stats[[joint]][[coord]]
      summary_data <- rbind(summary_data, data.frame(
        Joint       = joint,
        Coordinate  = coord,
        RMSE_Mean   = st$rmse_mean,
        RMSE_SD     = st$rmse_sd,
        ICC_Mean    = st$icc_mean,
        ICC_SD      = st$icc_sd,
        ICC_P_Value = st$icc_pvalue_mean,
        stringsAsFactors = FALSE
      ))
    }
  }
  
  writeData(wb, "Summary", summary_data, startRow=1, startCol=1)
  
  # 스타일 적용
  style_header <- createStyle(
    textDecoration="bold", halign="center", fgFill="#4F81BD", fontColour="white"
  )
  style_body <- createStyle(
    halign="center", border="TopBottom", borderColour="#4F81BD"
  )
  
  addStyle(wb, sheet="Summary", style_header, 
           rows=1, cols=1:ncol(summary_data), gridExpand=TRUE)
  addStyle(wb, sheet="Summary", style_body,
           rows=2:(nrow(summary_data)+1), cols=1:ncol(summary_data), gridExpand=TRUE)
  setColWidths(wb, sheet="Summary", cols=1:ncol(summary_data), widths="auto")
  
  # (2) Plots 시트
  addWorksheet(wb, "Plots")
  
  current_row <- 1
  if (!is.null(plot_image_paths)) {
    for (img_path in plot_image_paths) {
      if (file.exists(img_path)) {
        addImage(wb, sheet="Plots", file=img_path,
                 startRow=current_row, startCol=1,
                 width=25, height=15)
        current_row <- current_row + 25
      }
    }
  }
  
  saveWorkbook(wb, excel_file, overwrite=TRUE)
  cat("\n>> Final Excel with plots saved:", excel_file, "\n")
}

# ----------------------------------------------------
# 함수 8) main: 실제 실행 순서 (motion_type, subject_type)
# ----------------------------------------------------
main <- function(motion_type = NULL, subject_type = NULL) {
  cat(sprintf("\n[MAIN] Start: motion=%s, subject=%s\n", motion_type, subject_type))
  
  files_df <- find_edited_files(motion_type, subject_type)
  if (nrow(files_df) == 0) {
    stop("No files found matching criteria.")
  }
  
  # 결과 저장용
  results_list      <- list()  # (파일별) 조인트별 RMSE/ICC
  bland_altman_data <- list()  # (조인트, coord)에 대한 (marker_value, markerless_value) 모음
  
  for (i in seq_len(nrow(files_df))) {
    file_path <- files_df$file_path[i]
    data <- read_and_process_file(file_path)
    
    if (!is.null(data)) {
      # Bland-Altman용 데이터 축적
      for (joint in names(data)) {
        if (!(joint %in% names(bland_altman_data))) {
          bland_altman_data[[joint]] <- list()
        }
        for (coord in c("X","Y","Z","3D")) {
          if (!(coord %in% names(bland_altman_data[[joint]]))) {
            bland_altman_data[[joint]][[coord]] <- data.frame(
              marker_value = numeric(),
              markerless_value = numeric()
            )
          }
          mk <- data[[joint]]$marker_based[[coord]]
          ml <- data[[joint]]$markerless[[coord]]
          
          if (!is.null(mk) && !is.null(ml)) {
            valid_idx <- !is.na(mk) & !is.na(ml)
            bland_altman_data[[joint]][[coord]] <- rbind(
              bland_altman_data[[joint]][[coord]],
              data.frame(marker_value=mk[valid_idx], markerless_value=ml[valid_idx])
            )
          }
        }
      }
      
      # RMSE, ICC 분석
      file_results <- analyze_joints(data)
      results_list[[basename(file_path)]] <- file_results
    }
  }
  
  # (1) 여러 파일 결과를 통합( joint, coord별로 모두 모음 )
  combined_results <- list()
  for (file_name in names(results_list)) {
    file_res <- results_list[[file_name]]
    for (joint in names(file_res)) {
      if (! (joint %in% names(combined_results))) {
        combined_results[[joint]] <- list()
      }
      for (coord in names(file_res[[joint]])) {
        if (! (coord %in% names(combined_results[[joint]]))) {
          combined_results[[joint]][[coord]] <- list(
            icc_values  = c(),
            icc_pvalues = c(),
            rmse_values = c()
          )
        }
        combined_results[[joint]][[coord]]$icc_values <- c(
          combined_results[[joint]][[coord]]$icc_values,
          file_res[[joint]][[coord]]$icc_values
        )
        combined_results[[joint]][[coord]]$icc_pvalues <- c(
          combined_results[[joint]][[coord]]$icc_pvalues,
          file_res[[joint]][[coord]]$icc_pvalues
        )
        combined_results[[joint]][[coord]]$rmse_values <- c(
          combined_results[[joint]][[coord]]$rmse_values,
          file_res[[joint]][[coord]]$rmse_values
        )
      }
    }
  }
  
  # (2) 평균/표준편차 계산
  summary_stats <- list()
  for (joint in names(combined_results)) {
    summary_stats[[joint]] <- list()
    for (coord in names(combined_results[[joint]])) {
      icc_vals  <- combined_results[[joint]][[coord]]$icc_values
      icc_pvals <- combined_results[[joint]][[coord]]$icc_pvalues
      rmse_vals <- combined_results[[joint]][[coord]]$rmse_values
      
      summary_stats[[joint]][[coord]] <- list(
        icc_values     = icc_vals,
        icc_pvalues    = icc_pvals,
        rmse_values    = rmse_vals,
        icc_mean       = mean(icc_vals, na.rm=TRUE),
        icc_sd         = sd(icc_vals, na.rm=TRUE),
        icc_pvalue_mean= mean(icc_pvals, na.rm=TRUE),
        rmse_mean      = mean(rmse_vals, na.rm=TRUE),
        rmse_sd        = sd(rmse_vals, na.rm=TRUE)
      )
    }
  }
  
  # (3) 콘솔 출력
  cat("\n=== Summary ===\n")
  for (joint in names(summary_stats)) {
    cat(sprintf("\n[Joint] %s\n", joint))
    for (coord in names(summary_stats[[joint]])) {
      st <- summary_stats[[joint]][[coord]]
      cat(sprintf("  - %s : RMSE= %.2f ± %.2f, ICC= %.3f ± %.3f (p=%.4f)\n",
                  coord, st$rmse_mean, st$rmse_sd, st$icc_mean, st$icc_sd, st$icc_pvalue_mean))
    }
  }
  
  # (4) 그래프들 PNG 생성
  plot_paths <- c()
  
  # 4-A) Bland-Altman
  ba_paths <- create_bland_altman_plots(bland_altman_data, save_dir="visualization/bland_altman")
  plot_paths <- c(plot_paths, ba_paths)
  
  # 4-B) Combined RMSE 그래프(예시)
  comb_path <- create_combined_plot_example(summary_stats, output_dir="visualization/combined")
  plot_paths <- c(plot_paths, comb_path)
  
  # 필요 시 Time-series 그래프, coordinate별 trajectory 등
  # create_time_series_plots(...) 등으로 PNG저장 -> plot_paths에 추가
  
  # (5) Excel로 요약통계 + 모든 PNG 삽입
  excel_output_path <- sprintf("results/%s_%s_final.xlsx", 
                               ifelse(is.null(motion_type),"allMotion",motion_type),
                               ifelse(is.null(subject_type),"allSubject",subject_type))
  
  save_all_results_to_excel(
    summary_stats    = summary_stats,
    plot_image_paths = plot_paths,
    excel_file       = excel_output_path
  )
  
  cat("\n>> main() Done! See:", excel_output_path, "\n")
  
  return(list(summary_stats=summary_stats, bland_altman_data=bland_altman_data))
}

# ----------------------------------------------------
# (선택) 여러 subject 한 번에: analyze_all_subjects
# ----------------------------------------------------
analyze_all_subjects <- function(motion_type = "kicking") {
  
  # 예: 10명 목록
  subjects <- c("고승준", "김리언", "김준성", "김태형", 
                "방민식", "성기훈", "이준석", "장귀현",
                "김건희", "김훈민")
  
  # 결과를 담을 리스트
  all_subjects_results <- list()
  
  # ------------------------------------------------
  # 2) subjects를 순회하면서 main() 호출
  # ------------------------------------------------
  for (sbj in subjects) {
    cat(sprintf("\n=== Analyzing subject: %s ===\n", sbj))
    
    # main(motion_type, subject_type) 호출
    result <- main(motion_type = motion_type, subject_type = sbj)
    
    # 여기서 main()은 이미 “각 subject별” 그래프 + 엑셀 파일 저장 로직을 포함함.
    # => 예: results/kicking_김태형_final.xlsx 형태로 저장될 것.
    
    # 원하는 경우, result( summary_stats, bland_altman_data )를 모아서 저장
    all_subjects_results[[sbj]] <- result
    
    cat(sprintf("\n[Done subject: %s]\n", sbj))
  }
  
  # ------------------------------------------------
  # 3) (선택) 모든 subject 결과를 “종합”하여 추가 처리
  # ------------------------------------------------
  #  - 예: 모든 subject의 summary_stats를 합쳐서 한번 더 통계
  #  - 별도의 combined Excel 생성 등
  
  # 이 예시에서는 간단히 all_subjects_results만 반환
  return(all_subjects_results)