# 3.    RESULTS
# 3.5.  AGE AND ARTEFACT INFLUENCE ON ALGORITHM DEVELOPMENT 
# 3.5.1 ARTEFACT IMPACT ON ALGORITHM COMPLEXITY
library(dplyr)
library(tidyverse)
library(broom)
# library(lme4)
library(emmeans)
require(plyr)
library(lmerTest)
library(sjPlot)
library(car)

# preprocessing
df_u = read.csv('repositories/virtual-CAT-data-collection/src/output/tables/main/lme_dataset_unplugged.csv')
df_v = read.csv('repositories/virtual-CAT-data-collection/src/output/tables/main/lme_dataset_main.csv')
df_v = df_v[, !names(df_v) %in% "LOG_TIME"]
names(df_u)[names(df_u) == "ARMOS_GRADE"] <- "HARMOS_GRADE"

df_v = df_v %>%
  mutate(ARTEFACT = case_when(
    ARTEFACT_DIMENSION == 0 ~ "GF",
    ARTEFACT_DIMENSION == 1 ~ "G",
    ARTEFACT_DIMENSION == 2 ~ "PF",
    ARTEFACT_DIMENSION == 3 ~ "P",
    TRUE ~ "Other"  # Questa è la condizione di default se nessuna delle precedenti corrisponde
  ))


df_u = df_u %>%
  mutate(ARTEFACT = case_when(
    ARTEFACT_DIMENSION == 0 ~ "VSF",
    ARTEFACT_DIMENSION == 1 ~ "VS",
    ARTEFACT_DIMENSION == 2 ~ "V",
    TRUE ~ "Other"  # Questa è la condizione di default se nessuna delle precedenti corrisponde
  ))

df_v$ARTEFACT = factor(df_v$ARTEFACT, levels = c('GF', 'G', 'PF', 'P'))
df_u$ARTEFACT = factor(df_u$ARTEFACT, levels = c('VSF', 'VS', 'V'))

df_v$ARTEFACT = as.factor(df_v$ARTEFACT)

df_u$ARTEFACT = as.factor(df_u$ARTEFACT)

# anova
# is the artefact dimension a predictor of algorithmic skill in the unplugged dataset?
A_artefact_algorithm_df_u = aov(ALGORITHM_DIMENSION ~ ARTEFACT, data = df_u)
summary(A_artefact_algorithm_df_u)
# LM_artefact_algorithm_df_u = lm(ALGORITHM_DIMENSION ~ ARTEFACT, data = df_u)
# summary(LM_artefact_algorithm_df_u)
# is the artefact dimension a predictor of algorithmic skill in the virtual dataset?
A_artefact_algorithm_df_v = aov(ALGORITHM_DIMENSION ~ ARTEFACT, data = df_v)
summary(A_artefact_algorithm_df_v)
# LM_artefact_algorithm_df_v = lm(ALGORITHM_DIMENSION ~ ARTEFACT, data = df_v)
# summary(LM_artefact_algorithm_df_v)

# T-tests
t_test_result <- t.test(df_u$ALGORITHM_DIMENSION, df_v$ALGORITHM_DIMENSION)
cat("Risultati del test t:\n")
cat("Statistiche del test t =", t_test_result$statistic, "\n")
cat("Valore p =", t_test_result$p.value, "\n")
cat("\n")

# preprocessing: creation of a merged dataset
full_df = rbind(df_u, df_v)
full_df = full_df[, !names(full_df) %in% "ARTEFACT_DIMENSION"]

# LM_artefact_algorithm = lm(ALGORITHM_DIMENSION ~ ARTEFACT, data = full_df)
# summary(LM_artefact_algorithm)

# Tukey HSD test with BH (Benjamini-Hochberg) adjustment
A_artefact_algorithm = aov(ALGORITHM_DIMENSION ~ ARTEFACT, data = full_df)
tukey_result <- TukeyHSD(A_artefact_algorithm)
print(tukey_result)
p_values <- tukey_result$ARTEFACT[, "p adj"]
adjusted_p_values <- p.adjust(p_values, method = "BH")
print(adjusted_p_values)

# pairwise t-tests with Bonferroni correction
pairwise.t.test(full_df$ALGORITHM_DIMENSION, full_df$ARTEFACT, p.adjust.method = "bonferroni")


# chi-squared test of proportions
contingency_table_artefact <- full_df %>%
  group_by(ARTEFACT, ALGORITHM_DIMENSION) %>%
  dplyr::summarise(Count = n(), .groups = 'drop') %>%
  pivot_wider(names_from = ALGORITHM_DIMENSION, values_from = Count, values_fill = list(Count = 0))

contingency_table_artefact$Prop = contingency_table_artefact$`2` / (contingency_table_artefact$`2` + contingency_table_artefact$`1`+ contingency_table_artefact$`0`)

labels <- paste(contingency_table_artefact$ARTEFACT, sep = "_")

prop_test_results_artefact <- prop.test(contingency_table_artefact$`2`, (contingency_table_artefact$`1` + contingency_table_artefact$`2`+ contingency_table_artefact$`0`))

if(length(labels) == length(prop_test_results_artefact$estimate)) {
  names(prop_test_results_artefact$estimate) <- labels
}
proportions_df <- data.frame(
  Label = labels,
  Proportion = prop_test_results_artefact$estimate
)

print(prop_test_results_artefact)


# 3.    RESULTS
# 3.5.  AGE AND ARTEFACT INFLUENCE ON ALGORITHM DEVELOPMENT 
# 3.5.2 AGE-RELATED DEVELOPMENT IN ALGORITHMIC THINKING

# preprocessing
df_u$AGE_CATEGORY <- factor(df_u$AGE_CATEGORY, levels = c("From 3 to 6 years old", "From 7 to 9 years old",
                                                                "From 10 to 13 years old", "From 14 to 16 years old"))
df_v$AGE_CATEGORY <- factor(df_v$AGE_CATEGORY, levels = c("From 3 to 6 years old", "From 7 to 9 years old",
                                                                "From 10 to 13 years old", "From 14 to 16 years old"))
full_df$AGE_CATEGORY <- factor(full_df$AGE_CATEGORY, levels = c("From 3 to 6 years old", "From 7 to 9 years old",
                                                                "From 10 to 13 years old", "From 14 to 16 years old"))

df_v$AGE_CATEGORY = as.factor(df_v$AGE_CATEGORY)
df_v$ARTEFACT = as.factor(df_v$ARTEFACT)

df_u$AGE_CATEGORY = as.factor(df_u$AGE_CATEGORY)
df_u$ARTEFACT = as.factor(df_u$ARTEFACT)

# anova
# is the age a predictor of algorithmic skill in the unplugged dataset?
A_age_algorithm_df_u = aov(ALGORITHM_DIMENSION ~ AGE_CATEGORY, data = df_u)
summary(A_age_algorithm_df_u)
LM_age_algorithm_df_u = lm(ALGORITHM_DIMENSION ~ AGE_CATEGORY, data = df_u)
summary(LM_age_algorithm_df_u)
# is the age a predictor of algorithmic skill in the virtual dataset?
A_age_algorithm_df_v = aov(ALGORITHM_DIMENSION ~ AGE_CATEGORY, data = df_v)
summary(A_age_algorithm_df_v)
LM_age_algorithm_df_v = lm(ALGORITHM_DIMENSION ~ AGE_CATEGORY, data = df_v)
summary(LM_age_algorithm_df_v)

# anova with interaction terms
A_age_times_artefact_algorithm_df_u = aov(ALGORITHM_DIMENSION ~ AGE_CATEGORY * ARTEFACT, data = df_u)
summary(A_age_times_artefact_algorithm_df_u)
lm_summary_unplugged = summary(lm(ALGORITHM_DIMENSION ~ AGE_CATEGORY * ARTEFACT, data = df_u))
lm_summary_unplugged 

A_age_times_artefact_algorithm_df_v = aov(ALGORITHM_DIMENSION ~ AGE_CATEGORY * ARTEFACT, data = df_v)
summary(A_age_times_artefact_algorithm_df_v)
lm_summary_virtual = summary(lm(ALGORITHM_DIMENSION ~ AGE_CATEGORY * ARTEFACT, data = df_v))
lm_summary_virtual

#########New analysis after review #########
cols_to_convert = c('SCHEMA_ID', 'GENDER', 'SCHOOL_ID')
df_u[cols_to_convert] = lapply(df_u[cols_to_convert], factor)
df_v[cols_to_convert] = lapply(df_v[cols_to_convert], factor)


# is the Sex a predictor of algorithmic skill in the unplugged dataset?
A_sex_algorithm_df_u = aov(ALGORITHM_DIMENSION ~ GENDER, data = df_u)
summary(A_sex_algorithm_df_u)
# is the Sex a predictor of algorithmic skill in the virtual dataset?
A_sex_algorithm_df_v = aov(ALGORITHM_DIMENSION ~ GENDER, data = df_v)
summary(A_sex_algorithm_df_v)

# is the School a predictor of algorithmic skill in the unplugged dataset?
A_school_algorithm_df_u = aov(ALGORITHM_DIMENSION ~ SCHOOL_ID, data = df_u)
summary(A_school_algorithm_df_u)
# is the Sex a predictor of algorithmic skill in the virtual dataset?
A_school_algorithm_df_v = aov(ALGORITHM_DIMENSION ~ SCHOOL_ID, data = df_v)
summary(A_school_algorithm_df_v)

# is the Schema a predictor of algorithmic skill in the unplugged dataset?
A_schema_algorithm_df_u = aov(ALGORITHM_DIMENSION ~ SCHEMA_ID, data = df_u)
summary(A_schema_algorithm_df_u)
# is the Schema a predictor of algorithmic skill in the virtual dataset?
A_schema_algorithm_df_v = aov(ALGORITHM_DIMENSION ~ SCHEMA_ID, data = df_v)
summary(A_schema_algorithm_df_v)


# anova with interaction terms
summary(aov(ALGORITHM_DIMENSION ~ AGE_CATEGORY * GENDER, data = df_u))
summary(lm(ALGORITHM_DIMENSION ~ AGE_CATEGORY * GENDER, data = df_u))
summary(aov(ALGORITHM_DIMENSION ~ AGE_CATEGORY * GENDER, data = df_v))
summary(lm(ALGORITHM_DIMENSION ~ AGE_CATEGORY * GENDER, data = df_v))

#summary(aov(ALGORITHM_DIMENSION ~ AGE_CATEGORY * SCHOOL_ID, data = df_u))
#summary(lm(ALGORITHM_DIMENSION ~ AGE_CATEGORY * SCHOOL_ID, data = df_u))
#summary(aov(ALGORITHM_DIMENSION ~ AGE_CATEGORY * SCHOOL_ID, data = df_v))
#summary(lm(ALGORITHM_DIMENSION ~ AGE_CATEGORY * SCHOOL_ID, data = df_v))

# NOT IN THE PAPER
# summary(aov(ALGORITHM_DIMENSION ~ AGE_CATEGORY * SCHEMA_ID, data = df_u))
# summary(lm(ALGORITHM_DIMENSION ~ AGE_CATEGORY * SCHEMA_ID, data = df_u))
# NON-SIGNIFICATIVE
# summary(aov(ALGORITHM_DIMENSION ~ AGE_CATEGORY * SCHEMA_ID, data = df_v))
# summary(lm(ALGORITHM_DIMENSION ~ AGE_CATEGORY * SCHEMA_ID, data = df_v))

# TODO: INTERPRET
summary(aov(ALGORITHM_DIMENSION ~ ARTEFACT * GENDER, data = df_u))
summary(lm(ALGORITHM_DIMENSION ~ ARTEFACT * GENDER, data = df_u))
summary(aov(ALGORITHM_DIMENSION ~ ARTEFACT * GENDER, data = df_v))
summary(lm(ALGORITHM_DIMENSION ~ ARTEFACT * GENDER, data = df_v))

# NOT IN THE PAPER
# summary(aov(ALGORITHM_DIMENSION ~ ARTEFACT * SCHOOL_ID, data = df_u))
# summary(lm(ALGORITHM_DIMENSION ~ ARTEFACT * SCHOOL_ID, data = df_u))
# summary(aov(ALGORITHM_DIMENSION ~ ARTEFACT * SCHOOL_ID, data = df_v))
# summary(lm(ALGORITHM_DIMENSION ~ ARTEFACT * SCHOOL_ID, data = df_v))

# TODO: INTERPRET
summary(aov(ALGORITHM_DIMENSION ~ ARTEFACT * SCHEMA_ID, data = df_u))
summary(lm(ALGORITHM_DIMENSION ~ ARTEFACT * SCHEMA_ID, data = df_u))
summary(aov(ALGORITHM_DIMENSION ~ ARTEFACT * SCHEMA_ID, data = df_v))
summary(lm(ALGORITHM_DIMENSION ~ ARTEFACT * SCHEMA_ID, data = df_v))

# NOT IN THE PAPER
# summary(aov(ALGORITHM_DIMENSION ~ GENDER * SCHOOL_ID, data = df_u))
# summary(lm(ALGORITHM_DIMENSION ~ GENDER * SCHOOL_ID, data = df_u))
# summary(aov(ALGORITHM_DIMENSION ~ GENDER * SCHOOL_ID, data = df_v))
# summary(lm(ALGORITHM_DIMENSION ~ GENDER * SCHOOL_ID, data = df_v))

# NON-SIGNIFICATIVE
# summary(aov(ALGORITHM_DIMENSION ~ GENDER * SCHEMA_ID, data = df_u))
# summary(lm(ALGORITHM_DIMENSION ~ GENDER * SCHEMA_ID, data = df_u))
# summary(aov(ALGORITHM_DIMENSION ~ GENDER * SCHEMA_ID, data = df_v))
# summary(lm(ALGORITHM_DIMENSION ~ GENDER * SCHEMA_ID, data = df_v))

# NOT IN THE PAPER
# summary(aov(ALGORITHM_DIMENSION ~ SCHOOL_ID * SCHEMA_ID, data = df_u))
# summary(lm(ALGORITHM_DIMENSION ~ SCHOOL_ID * SCHEMA_ID, data = df_u))
# summary(aov(ALGORITHM_DIMENSION ~ SCHOOL_ID * SCHEMA_ID, data = df_v))
# summary(lm(ALGORITHM_DIMENSION ~ SCHOOL_ID * SCHEMA_ID, data = df_v))

######### End of new anal #########

em_unplugged <- emmeans(A_age_times_artefact_algorithm_df_u, ~ AGE_CATEGORY * ARTEFACT)
summary(em_unplugged)
pairs(em_unplugged)
pairs(em_unplugged, adjust = "tukey")


em_virtual <- emmeans(A_age_times_artefact_algorithm_df_v, ~ AGE_CATEGORY * ARTEFACT)
summary(em_virtual)
pairs(em_virtual)
pairs(em_virtual, adjust = "tukey")



# summary(aov(ALGORITHM_DIMENSION ~ AGE_CATEGORY * ARTEFACT, data = full_df))
lm_summary = summary(lm(ALGORITHM_DIMENSION ~ AGE_CATEGORY * ARTEFACT, data = full_df))
lm_summary

# anova_result_full <- aov(ALGORITHM_DIMENSION ~ DOMAIN + ARTEFACT * AGE_CATEGORY, data = full_df)
# summary(anova_result_full)
# summary(lm(ALGORITHM_DIMENSION ~ DOMAIN + ARTEFACT * AGE_CATEGORY, data = full_df))


# # Tukey HSD test with BH (Benjamini-Hochberg) adjustment
# A_age_times_artefact_algorithm = aov(ALGORITHM_DIMENSION ~ AGE_CATEGORY * ARTEFACT, data = full_df)
# tukey_result <- TukeyHSD(A_age_times_artefact_algorithm)
# print(tukey_result)
# p_values <- tukey_result$`AGE_CATEGORY:ARTEFACT`[, "p adj"]
# adjusted_p_values <- p.adjust(p_values, method = "BH")
# print(adjusted_p_values)

# # pairwise t-tests with Bonferroni correction
# pairwise.t.test(full_df$ALGORITHM_DIMENSION, full_df$AGE_CATEGORY, p.adjust.method = "bonferroni")
# pairwise_comp <- pairwise.t.test(full_df$ALGORITHM_DIMENSION, 
#                                  interaction(full_df$AGE_CATEGORY, full_df$ARTEFACT),
#                                  p.adjust.method = "bonferroni")
# 
# print(pairwise_comp)

# # Calculate Cohen's
# cohen_d_u <- lm_summary_unplugged$coefficients[, "Estimate"] / lm_summary_unplugged$coefficients[, "Std. Error"]
# print(cohen_d_u)
# cohen_d_v <- lm_summary_virtual$coefficients[, "Estimate"] / lm_summary_virtual$coefficients[, "Std. Error"]
# print(cohen_d_v)
# 
# lm_summary = summary(lm(ALGORITHM_DIMENSION ~ DOMAIN * AGE_CATEGORY * ARTEFACT, data = full_df))
# cohen_d_v <- lm_summary$coefficients[, "Estimate"] / lm_summary$coefficients[, "Std. Error"]
# print(cohen_d_v)

# Definisci le categorie di età
age_categories <- c("From 3 to 6 years old", "From 7 to 9 years old", "From 10 to 13 years old", "From 14 to 16 years old")

# Soglia minima per il numero di casi
min_cases_threshold <- 5  # Modifica la soglia a tua discrezione

# Inizializza una lista per memorizzare i risultati
prop_test_results <- list()

# Itera attraverso le categorie di età
for (age_category in age_categories) {
  # Filtra il dataframe per la categoria di età corrente
  df_age_category <- full_df %>%
    filter(AGE_CATEGORY == age_category)
  
  # Crea una tabella di contingenza
  contingency_table <- df_age_category %>%
    group_by(ARTEFACT, ALGORITHM_DIMENSION) %>%
    dplyr::summarise(Count = n(), .groups = 'drop') %>%
    pivot_wider(names_from = ALGORITHM_DIMENSION, values_from = Count, values_fill = list(Count = 0))
  
  # Filtra le righe con `2` diverso da 0 e con un numero minimo di casi
  contingency_table <- contingency_table %>%
    filter(`2` != 0 & (`1` + `2` + `0`) >= min_cases_threshold)
  
  # Esegui il test di proporzioni solo se ci sono abbastanza casi
  if (nrow(contingency_table) > 0) {
    prop_test_result <- prop.test(contingency_table$`2`, contingency_table$`1` + contingency_table$`2` + contingency_table$`0`)
    
    # Assegna le labels corrette
    labels <- paste(contingency_table$ARTEFACT, sep = "_")
    
    if (length(labels) == length(prop_test_result$estimate)) {
      names(prop_test_result$estimate) <- labels
    }
    
    # Aggiungi il risultato alla lista
    prop_test_results[[age_category]] <- prop_test_result
  } else {
    # Se ci sono troppo pochi casi, assegna "Non calcolabile" o "N/A"
    prop_test_results[[age_category]] <- "Non calcolabile"
  }
  
}

print(prop_test_results)
 
 ######
 
 # Prepara i dati per il dominio unplugged
 unplugged_df <- full_df %>%
   filter(DOMAIN == "Unplugged") %>%
   group_by(AGE_CATEGORY) %>%
  dplyr::summarise(Success = sum(ALGORITHM_DIMENSION == 2), Total = n(), .groups = 'drop') %>%
   mutate(Proportion = Success / Total)

 # -------------------------------------------------------------------------


 labels <- paste(unplugged_df$AGE_CATEGORY, sep = "_")

 # Test di proporzioni per il dominio unplugged
 prop_test_unplugged <- prop.test(unplugged_df$Success, unplugged_df$Total)
 if(length(labels) == length(prop_test_unplugged$estimate)) {
   names(prop_test_unplugged$estimate) <- labels
 }

 # Prepara i dati per il dominio virtual
 virtual_df <- full_df %>%
   filter(DOMAIN == "Virtual") %>%
   group_by(AGE_CATEGORY) %>%
   dplyr::summarise(Success = sum(ALGORITHM_DIMENSION == 2), Total = n(), .groups = 'drop') %>%
   mutate(Proportion = Success / Total)

 labels <- paste(unplugged_df$AGE_CATEGORY, sep = "_")

 # Test di proporzioni per il dominio virtual
 prop_test_virtual <- prop.test(virtual_df$Success, virtual_df$Total)
 if(length(labels) == length(prop_test_virtual$estimate)) {
   names(prop_test_virtual$estimate) <- labels
 }


 # Stampa i risultati
 print(prop_test_unplugged)
 print(prop_test_virtual)


#####
 
df_v$ALGORITHM_DIMENSION = as.factor(df_v$ALGORITHM_DIMENSION)
df_v$AGE_CATEGORY = as.factor(df_v$AGE_CATEGORY)
df_v$ARTEFACT = as.factor(df_v$ARTEFACT)

df_u$ALGORITHM_DIMENSION = as.factor(df_u$ALGORITHM_DIMENSION)
df_u$AGE_CATEGORY = as.factor(df_u$AGE_CATEGORY)
df_u$ARTEFACT = as.factor(df_u$ARTEFACT)


full_df$ALGORITHM_DIMENSION = as.factor(full_df$ALGORITHM_DIMENSION)
full_df$AGE_CATEGORY = as.factor(full_df$AGE_CATEGORY)
full_df$ARTEFACT = as.factor(full_df$ARTEFACT)


full_df$Success = ifelse(full_df$ALGORITHM_DIMENSION == 2, 1, 0)

model <- glm(Success ~ AGE_CATEGORY * ARTEFACT, data = full_df, family = binomial)
summary(model)

anova_result <- anova(model)
print(anova_result)
print(n=30,tidy(model))



# Logistic regression model
model <- glm(ALGORITHM_DIMENSION ~ AGE_CATEGORY * ARTEFACT, data = df_v, family = "binomial")

# Summary of the model
summary(model)

# For a detailed summary including p-values
print(n=30,tidy(model))



