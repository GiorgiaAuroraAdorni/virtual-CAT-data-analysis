require(plyr)
library(lmerTest)
library(sjPlot)
library(ggplot2)
library(car)
library(dplyr)
library(lattice)

data_virtual = read.csv('repositories/virtual-CAT-data-collection/src/output/tables/main/lme_dataset_main.csv')
data_virtual$SESSION_GRADE = mapvalues(data_virtual$SESSION_ID, 
                                       from = c('1', '2', '3', '4', '5', '6', '7', '8','9'), 
                                       to = c('3V (4H)', '1V (0H, 1H, 2H)', '5V (8H)', '6V (10H)', 
                                              '7V (10H)','8V (11H)', '2V (2H)', '9V (11H)', '4V (6H)'))
data_virtual$SESSION_GRADE = factor(data_virtual$SESSION_GRADE,
                                    levels = c('1V (0H, 1H, 2H)', '2V (2H)', '3V (4H)', '4V (6H)', 
                                               '5V (8H)', '6V (10H)', '7V (10H)', '8V (11H)', '9V (11H)'))

data_virtual$SCHOOL_ID = factor(data_virtual$SCHOOL_ID, levels = c('A', 'D', 'E', 'F', 'G'))

data_virtual_old = data_virtual # This is used for variance comparison

virtual_student_id = paste0("V", data_virtual$STUDENT_ID)
virtual_student_id_unique = unique(virtual_student_id)
data_virtual$STUDENT_ID = factor(virtual_student_id, levels=virtual_student_id_unique)

cols_to_convert = c('SCHEMA_ID', 'GENDER', 'SESSION_GRADE', 'HARMOS_GRADE', 'SCHOOL_ID', 'CANTON_NAME', 'DOMAIN')
data_virtual[cols_to_convert] = lapply(data_virtual[cols_to_convert], factor)

# Rescale the CAT_SCORE between 0 and 1
rescale_virtual_CAT_SCORE = function(grade, score) {
  if (grade %in% c(0, 1, 2)) {
    # If ARMOS_GRADE is 0, 1, or 2, rescale with max value of 3
    return(score / 3*5)
  } else {
    # If ARMOS_GRADE is higher than 2, rescale with max value of 5
    return(score / 5*5)
  }
}
data_virtual$rescaled_CAT_SCORE = mapply(rescale_virtual_CAT_SCORE, data_virtual$HARMOS_GRADE, data_virtual$CAT_SCORE)


# 3.  RESULTS
# 3.6 LINEAR MIXED MODEL ASSESSMENT OF STUDENT PERFORMANCE
M3 = lmer('rescaled_CAT_SCORE ~ (1|SCHEMA_ID) + (1+GENDER|SCHOOL_ID) + (1|SESSION_GRADE) + (1|STUDENT_ID)', data_virtual)
summary(M3)
ranova(M3, reduce.terms = TRUE)

plots = plot_model(M3, type = "re", title=' ')
plots[[1]] = plots[[1]] + labs(x = "Student", y = "CAT-score") + theme_minimal()
plots[[2]] = plots[[2]] + labs(x = "Schema", y = "CAT-score") + theme_minimal()
plots[[3]] = plots[[3]] + labs(x = "Session-Grade", y = "CAT-score") + theme_minimal()
plots[[4]] = plots[[4]] + labs(x = "School", y = "CAT-score") + theme_minimal()

pdf(file = "repositories/virtual-CAT-data-collection/src/output/images/main/pdf/model/virtual_re_student.pdf")
plots[[1]]
dev.off() 

pdf(file = "repositories/virtual-CAT-data-collection/src/output/images/main/pdf/model/virtual_re_schema.pdf")
plots[[2]]
dev.off() 

pdf(file = "repositories/virtual-CAT-data-collection/src/output/images/main/pdf/model/virtual_re_sessiongrade.pdf")
plots[[3]]
dev.off() 

pdf(file = "repositories/virtual-CAT-data-collection/src/output/images/main/pdf/model/virtual_re_school.pdf")
plots[[4]]
dev.off() 


model_data = get_model_data(M3, type="re")
student_data = model_data[[1]]
merged_data = merge(student_data, data_virtual_old, by.x = "row.names", by.y = "STUDENT_ID")

student_data$school = factor(student_data$p.label)
pdf(file = "repositories/virtual-CAT-data-collection/src/output/images/main/pdf/model/virtual_student_var.pdf")
ggplot(merged_data, aes(x = factor(SCHOOL_ID), y = estimate)) +
  geom_boxplot() +
  labs(x = "School", y = "Estimate (CAT-score Variance)") +
  theme_minimal()
dev.off() 

leveneTest(rescaled_CAT_SCORE ~ SCHOOL_ID, data = data_virtual)


# 3.     RESULTS
# 3.6.   LINEAR MIXED MODEL ASSESSMENT OF STUDENT PERFORMANCE
# 3.6.5. Effect of task completion time on performance
time_data_virtual = data_virtual
decile_bounds = quantile(time_data_virtual$LOG_TIME, probs = seq(0, 1, by = 0.1), na.rm = TRUE)

decile_labels = sapply(1:10, function(i) {
  if (i == 1) {
    return(paste("[", decile_bounds[i], "-", decile_bounds[i+1], "]"))
  } else {
    return(paste("(", decile_bounds[i], "-", decile_bounds[i+1], "]"))
  }
})

time_data_virtual = time_data_virtual %>%
  mutate(
    TASK_COMPLETION_TIME = case_when(
      ntile(LOG_TIME, 10) == 1 ~  decile_labels[1],
      ntile(LOG_TIME, 10) == 2 ~  decile_labels[2],
      ntile(LOG_TIME, 10) == 3 ~  decile_labels[3],
      ntile(LOG_TIME, 10) == 4 ~  decile_labels[4],
      ntile(LOG_TIME, 10) == 5 ~  decile_labels[5],
      ntile(LOG_TIME, 10) == 6 ~  decile_labels[6],
      ntile(LOG_TIME, 10) == 7 ~  decile_labels[7],
      ntile(LOG_TIME, 10) == 8 ~  decile_labels[8],
      ntile(LOG_TIME, 10) == 9 ~  decile_labels[9],
      ntile(LOG_TIME, 10) == 10 ~ decile_labels[10],
      TRUE ~ NA_character_
    )
  )

time_data_virtual$TASK_COMPLETION_TIME = factor(time_data_virtual$TASK_COMPLETION_TIME, levels = decile_labels)

virtual_model = lmer('rescaled_CAT_SCORE ~ (1|SCHEMA_ID) + (1+GENDER|SCHOOL_ID) + (1|SESSION_GRADE) + (1|STUDENT_ID) +(1|TASK_COMPLETION_TIME)', time_data_virtual)

plots = plot_model(virtual_model, type = "re", title=' ')

# For the first plot
plots[[1]] = plots[[1]] + labs(x = "Completion time range", y = "CAT-score") + theme_minimal()
plots[[2]] = plots[[2]] + labs(x = "Student", y = "CAT-score") + theme_minimal()
plots[[3]] = plots[[3]] + labs(x = "Schema", y = "CAT-score") + theme_minimal()
plots[[4]] = plots[[4]] + labs(x = "Session-Grade", y = "CAT-score") + theme_minimal()
plots[[5]] = plots[[5]] + labs(x = "School", y = "CAT-score") + theme_minimal()

pdf(file = "repositories/virtual-CAT-data-collection/src/output/images/main/pdf/model/re_student_domain.pdf")
plots[[1]]
dev.off() 

pdf(file = "repositories/virtual-CAT-data-collection/src/output/images/main/pdf/model/re_schema_domain.pdf")
plots[[2]]
dev.off() 

pdf(file = "repositories/virtual-CAT-data-collection/src/output/images/main/pdf/model/re_time_domain.pdf")
plots[[3]]
dev.off() 

pdf(file = "repositories/virtual-CAT-data-collection/src/output/images/main/pdf/model/re_session_domain.pdf")
plots[[4]]
dev.off() 

pdf(file = "repositories/virtual-CAT-data-collection/src/output/images/main/pdf/model/re_school.pdf")
plots[[5]]
dev.off() 
