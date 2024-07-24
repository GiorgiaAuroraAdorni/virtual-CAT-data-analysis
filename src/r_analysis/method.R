require(plyr)
library(lmerTest)
library(sjPlot)
library(ggplot2)
library(car)
library(dplyr)

# 2.     METHOD
# 2.4.   DATA ANALYSIS 
# 2.4.6. LINEAR MIXED MODEL ASSESSMENT OF STUDENT PERFORMANCE
#        MODEL SELECTION AND REFINEMENT

data_virtual = read.csv('repositories/virtual-CAT-data-collection/src/output/tables/main/lme_dataset_main.csv')
data_virtual$SESSION_GRADE = mapvalues(data_virtual$SESSION_ID, 
                                       from = c('1', '2', '3', '4', '5', '6', '7', '8','9'), 
                                       to = c('3V (4H)', '1V (0H, 1H, 2H)', '5V (8H)', '6V (10H)', 
                                              '7V (10H)','8V (11H)', '2V (2H)', '9V (11H)', '4V (6H)'))
data_virtual$SESSION_GRADE = factor(data_virtual$SESSION_GRADE,
                                    levels = c('1V (0H, 1H, 2H)', '2V (2H)', '3V (4H)', '4V (6H)', 
                                               '5V (8H)', '6V (10H)', '7V (10H)', '8V (11H)', '9V (11H)'))
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


M0 = lmer('rescaled_CAT_SCORE ~ GENDER + CANTON_NAME + (1|SCHEMA_ID) + (1|SCHOOL_ID) + (1|SESSION_GRADE) + (1|STUDENT_ID)', data_virtual)
summary(M0)

# BASELINE MODEL Y/N CANTON
M1 = lmer('rescaled_CAT_SCORE ~ GENDER + (1|SCHEMA_ID) + (1|SCHOOL_ID) + (1|SESSION_GRADE) + (1|STUDENT_ID)', data_virtual)
summary(M1)
anova(M0, M1)
# BASELINE MODEL Y/N GENDER
M2 = lmer('rescaled_CAT_SCORE ~ (1|SCHEMA_ID) + (1|SCHOOL_ID) + (1|SESSION_GRADE) + (1|STUDENT_ID)', data_virtual)
summary(M2)
M3 = lmer('rescaled_CAT_SCORE ~ (1|SCHEMA_ID) + (1+GENDER|SCHOOL_ID) + (1|SESSION_GRADE) + (1|STUDENT_ID)', data_virtual)
summary(M3)
anova(M1, M2, M3)

