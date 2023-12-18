library(rsample)
library(dplyr)
library(ggplot2)
library(h2o)
library(foreign)
library(caret)
library(tidymodels)

setwd("C:\\Users\\augus\\Documents\\GitHub\\Fork\\phishing-detection")

data <- read.arff("dataSet.arff")

set.seed(2023)
split <- initial_split(data, prop=.7, strata = "Result")
train <- training(split)
test <- testing(split)

features <- setdiff(names(data), "Result")

# Convert features to numerical (excluding 'Result')
data <- data %>%
  mutate_if(names(data) %in% features, as.numeric)

#h2o implementation
#h2o.shutdown()
h2o.no_progress()
h2o.init()

y <- "Result"

preprocess <- preProcess(train, method = c("BoxCox", "center", "scale", "pca"))
train_pp   <- predict(preprocess, train)
test_pp    <- predict(preprocess, test)

train_pp.h2o <- train_pp %>%
  mutate_if(is.factor, factor, ordered = FALSE) %>%
  as.h2o()

test_pp.h2o <- test_pp %>%
  mutate_if(is.factor, factor, ordered = FALSE) %>%
  as.h2o()

x_h2o <- setdiff(names(train_pp), y)

nb_model <- h2o.naiveBayes(x = x_h2o, y = y, training_frame = train_pp.h2o)
predictions <- as.vector(h2o.predict(nb_model, newdata = test_pp.h2o))


true_labels <- as.vector(test_pp.h2o$Result)


accuracy <- sum(predictions == true_labels) / length(true_labels)
cat("Accuracy:", accuracy, "\n")

param_grid <- list(
  laplace = seq(0, 5, by = 0.5)
)

grid <- h2o.grid(
  algorithm="naivebayes",
  grid_id="nb_grid",
  x=x_h2o,
  y=y,
  training_frame=train_pp.h2o,
  nfolds=10,
  hyper_params=param_grid
)
sorted_grid <- h2o.getGrid("nb_grid", sort_by = "accuracy", decreasing = TRUE)
sorted_grid
best_h2o_model <- sorted_grid@model_ids[[1]]
best_model <- h2o.getModel(best_h2o_model)

h2o.saveModel(best_model, path = "./", force = TRUE)

auc <- h2o.auc(best_model, train = TRUE, valid = FALSE)
cat("AUC-ROC:", auc, "\n")
p_value <- 2 * (1 - auc)
cat("P-value:", p_value, "\n")

y_pred <- as.vector(h2o.predict(best_model, test_pp.h2o)$predict)
y_true <- test$Result
y_pred <- as.factor(y_pred)
y_true <- as.factor(y_true)
y_prob <- as.vector(h2o.predict(best_model, test_pp.h2o)$p1)

pr_auc <- h2o.aucpr(best_model, train = TRUE, valid = FALSE)
cat("AUC-PR:", pr_auc, "\n")


conf_matrix <- h2o.confusionMatrix(best_model)
tp <- as.numeric(conf_matrix[2, 2])
fp <- as.numeric(conf_matrix[1, 2])
fn <- as.numeric(conf_matrix[2,1])
precision <- tp / (tp + fp)
recall <- tp / (tp+fn)
f1_score <-  2 * (precision * recall) / (precision + recall)
cat("F1 Score:", f1_score, "\n")

fpr <- h2o.performance(best_model, xval = TRUE) %>% h2o.fpr() %>% .[['fpr']]
tpr <- h2o.performance(best_model, xval = TRUE) %>% h2o.tpr() %>% .[['tpr']]
auc <- h2o.auc(best_model, xval = TRUE)
data.frame(fpr = fpr, tpr = tpr) %>%
  ggplot(aes(fpr, tpr) ) +
  geom_line() +
  ggtitle( sprintf('AUC: %f', auc) ) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "blue") +
  xlim(c(0, 1)) +
  ylim(c(0, 1))

fnr <- 1 - tpr
data.frame(fpr, fnr) %>%
  ggplot(aes(fpr, fnr)) +
  geom_line() +
  ggtitle("Detection Error Tradeoff (DET) Curve") +
  xlab("False Positive Rate (FPR)") +
  ylab("False Negative Rate (FNR)")

conf_matrix <- caret::confusionMatrix(y_true, y_pred)


accuracy <- sum(diag(conf_matrix$table)) / sum(conf_matrix$table)
cat("Accuracy:", accuracy, "\n")



################### CONFUSION MATRIX ######################
library(tidyr)
conf_matrix_df <- as.data.frame.matrix(conf_matrix$table)
conf_matrix_df_norm <- prop.table(as.matrix(conf_matrix_df), margin = 1)
conf_matrix_df_norm <- as.data.frame.matrix(conf_matrix_df_norm)
conf_matrix_df_norm <- cbind(TClass = rownames(conf_matrix_df_norm), as.data.frame(conf_matrix_df_norm))
df_norm <- pivot_longer(conf_matrix_df_norm, cols = -TClass, names_to = "PClass", values_to = "value")

ggplot(data = df_norm, mapping = aes(x = PClass, y = TClass)) +
  geom_tile(aes(fill = value), colour = "white") +
  geom_text(aes(label = sprintf("%.2f", value)), vjust = 1) +
  scale_fill_gradient(low = "blue", high = "red") +
  theme_bw() + theme(legend.position = "none") +
  labs(title = "Normalized Confusion Matrix", x = "Predicted Class", y = "True Class")
##########################################################


################### PR CURVE ######################
library(PRROC)
weights_class0 <- as.numeric(test$Result == 1)
pr_result_with_curve <- pr.curve( y_prob, weights.class0 = weights_class0, curve = TRUE)

plot(pr_result_with_curve$curve, type = 'l', col = 'blue', lwd = 2,
     xlab = 'Recall', ylab = 'Precision', main = 'Precision-Recall Curve')

####################################################


#Permutation importance
accuracy_before_permutation <- confusionMatrix(y_true, y_pred)$overall["Accuracy"]
importance_values <- numeric(length(features))
for (i in seq_along(features)) {
  permuted_data <- test_pp
  permuted_data[[features[i]]] <- sample(permuted_data[[features[i]]])

  # Make predictions on the permuted data
  y_permuted <- as.vector(h2o.predict(best_model, as.h2o(permuted_data))$predict)

  # Calculate accuracy after permutation
  accuracy_after_permutation <- sum(y_permuted == y_true) / length(y_true)

  # Calculate importance value
  importance_values[i] <- accuracy_before_permutation - accuracy_after_permutation
}
importance_df <- data.frame(feature = features, importance = importance_values)
ranked_features <- importance_df[order(importance_values, decreasing = TRUE), ]


importance_df <- data.frame(feature = ranked_features, importance = importance_values)

ggplot(ranked_features, aes(x = reorder(feature, -importance), y = importance)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Permutation Importance of Features",
       x = "Features",
       y = "Importance") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
