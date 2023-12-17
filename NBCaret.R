library(caret)
features <- setdiff(names(train), "Result")
x <- train[, features]
y <- train$Result

train_control <- trainControl(
  method="cv",
  number=10
)

nb.model <- train(x=x, y=y, method="nb", trControl=train_control)
confusionMatrix(nb.model)

grid_search <- expand.grid(
  usekernel=c(TRUE,FALSE),
  fL=0:5,
  adjust=seq(0,5,by=1)
)

nb.best_model <-train(x=x,y=y, method="nb", trControl=train_control, tuneGrid=grid_search, prePoc=c("BoxCox", "center", "scale", "pca"))
nb.best_model$results %>% 
  top_n(5, wt = Accuracy) %>%
  arrange(desc(Accuracy))
