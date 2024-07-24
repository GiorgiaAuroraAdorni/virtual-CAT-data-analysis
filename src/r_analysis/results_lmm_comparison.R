require(plyr)
library(lmerTest)
library(sjPlot)
library(ggplot2)
library(car)
library(dplyr)

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


data_virtual = data_virtual[, !names(data_virtual) %in% "LOG_TIME"]

data_unplugged = read.csv('repositories/virtual-CAT-data-collection/src/output/tables/main/lme_dataset_unplugged.csv')
data_unplugged$SESSION_GRADE = mapvalues(data_unplugged$SESSION_ID, 
                                         from = c('1', '2', '3', '4', '5', '6', '7', '8'), 
                                         to = c('1U (0H, 1H, 2H)', '5U (9H)', '7U (11H)', '6U (10H)', 
                                                '8U (11H)', '3U (5H)', '4U (7H)', '2U (3H)'))
data_unplugged$SESSION_GRADE = factor(data_unplugged$SESSION_GRADE, 
                                      levels = c('1U (0H, 1H, 2H)', '2U (3H)', '3U (5H)', '4U (7H)', 
                                                 '5U (9H)', '6U (10H)', '7U (11H)', '8U (11H)'))
data_unplugged$SCHOOL_ID = factor(data_unplugged$SCHOOL_ID, levels = c('A', 'B', 'C'))


unplugged_student_id = paste0("U", data_unplugged$STUDENT_ID)
unplugged_student_id_unique = unique(unplugged_student_id)
data_unplugged$STUDENT_ID = factor(unplugged_student_id, levels=unplugged_student_id_unique)

names(data_unplugged)[names(data_unplugged) == "ARMOS_GRADE"] <- "HARMOS_GRADE"
data_unplugged[cols_to_convert] = lapply(data_unplugged[cols_to_convert], factor)

rescale_unplugged_CAT_SCORE = function(grade, score) {
  return(score / 4*5)
}
data_unplugged$rescaled_CAT_SCORE = mapply(rescale_unplugged_CAT_SCORE, data_unplugged$HARMOS_GRADE, data_unplugged$CAT_SCORE)




# 3.     RESULTS
# 3.6.   LINEAR MIXED MODEL ASSESSMENT OF STUDENT PERFORMANCE
# 3.6.6. Student performance dynamics in virtual and unplugged settings

M3_unplugged = lmer('rescaled_CAT_SCORE ~ (1|SCHEMA_ID) + (1+GENDER|SCHOOL_ID) + (1|SESSION_GRADE) + (1|STUDENT_ID)', data_unplugged)
summary(M3_unplugged)
ranova(M3_unplugged)

# FULL DATASET
df = rbind(data_unplugged, data_virtual)
df$SESSION_GRADE = factor(df$SESSION_GRADE,
                          levels = c('1U (0H, 1H, 2H)', '1V (0H, 1H, 2H)', '2V (2H)', '2U (3H)',
                                     '3V (4H)', '3U (5H)', '4V (6H)', '4U (7H)', '5V (8H)',
                                     '5U (9H)', '6V (10H)', '6U (10H)', '7V (10H)', '7U (11H)', 
                                     '8V (11H)', '8U (11H)', '9V (11H)'))

df$SCHOOL_ID = factor(df$SCHOOL_ID, levels = c('A', 'B', 'C', 'D', 'E', 'F', 'G'))

M3_full_domain = lmer('rescaled_CAT_SCORE ~ DOMAIN + (1|SCHEMA_ID) + (1+GENDER|SCHOOL_ID) + (1|SESSION_GRADE) + (1|STUDENT_ID)', df)
summary(M3_full_domain)

M3_full = lmer('rescaled_CAT_SCORE ~ (1|SCHEMA_ID) + (1+GENDER|SCHOOL_ID) + (1|SESSION_GRADE) + (1|STUDENT_ID)', df)
anova(M3_full_domain,M3_full)
ranova(M3_full)  


M5_full_domain = lmer('rescaled_CAT_SCORE ~ DOMAIN + GENDER + (1|SCHEMA_ID) + (1|SCHOOL_ID) + (1|SESSION_GRADE) + (1|STUDENT_ID)', df)
M4_full_domain = lmer('rescaled_CAT_SCORE ~ DOMAIN + (1|SCHEMA_ID) + (1|SCHOOL_ID) + (1|SESSION_GRADE) + (1|STUDENT_ID)', df)

anova(M3_full_domain,M4_full_domain,M5_full_domain)
anova(M4_full_domain,M5_full_domain)


M5_full = lmer('rescaled_CAT_SCORE ~ GENDER + (1|SCHEMA_ID) + (1|SCHOOL_ID) + (1|SESSION_GRADE) + (1|STUDENT_ID)', df)
M4_full = lmer('rescaled_CAT_SCORE ~ (1|SCHEMA_ID) + (1|SCHOOL_ID) + (1|SESSION_GRADE) + (1|STUDENT_ID)', df)

anova(M3_full,M4_full,M5_full)
anova(M4_full,M5_full)


##  UNPLUGGED
data_unplugged$AGE_CATEGORY <- as.factor(data_unplugged$AGE_CATEGORY)
chi_square_result <- chisq.test(data_unplugged$CAT_SCORE, data_unplugged$AGE_CATEGORY)
print(chi_square_result)


# Fit a one-way ANOVA model and perform Tukey's HSD test with Benjamini-Hochberg correction
model <- aov(CAT_SCORE ~ AGE_CATEGORY, data = data_unplugged)
tukey_results <- TukeyHSD(model, "AGE_CATEGORY", conf.level = 0.95)
print(tukey_results)

##  VIRTUAL

data_virtual$AGE_CATEGORY <- as.factor(data_virtual$AGE_CATEGORY)
chi_square_result <- chisq.test(data_virtual$CAT_SCORE, data_virtual$AGE_CATEGORY)
print(chi_square_result)


# Fit a one-way ANOVA model and perform Tukey's HSD test with Benjamini-Hochberg correction
model <- aov(CAT_SCORE ~ AGE_CATEGORY, data = data_virtual)
tukey_results <- TukeyHSD(model, "AGE_CATEGORY", conf.level = 0.95)
print(tukey_results)
