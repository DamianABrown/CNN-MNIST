rm(list=ls())
library(keras)

batch.size <- 128
num.classes <- 10
epochs <- 12

#Establishing image dimensions "pixels"
img.rows <- 28
img.cols <- 28

#Importing data and splitting into training and testing sets
mnist <-dataset_mnist()
x.train <- mnist$train$x
y.train <- mnist$train$y
x.test <- mnist$test$x
y.test <- mnist$test$y

#Redifining dimensions of data to pass through CNN
x.train <- array_reshape(x.train, c(nrow(x.train), img.rows, img.cols, 1))
x.test <- array_reshape(x.test, c(nrow(x.test), img.rows, img.cols, 1))
input.shape <- c(img.rows, img.cols, 1)

#Changing values of pixels from 0 to 1 rather than 0 to 255.
x.train <- x.train/255
x.test <- x.test/255

#Convert class vectors to binary class matrices
y.train <- to_categorical(y.train, num.classes)
y.test <- to_categorical(y.test, num.classes)

cat('x_train_shape:', dim(x.train), '\n')
cat(nrow(x.train), 'train samples\n')
cat(nrow(x.test), 'test samples\n')

#Defining Model
model <- keras_model_sequential()

#Building CNN
model %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,4), activation = 'relu',
                input_shape = input.shape) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(2,2), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = num.classes, activation = 'softmax')

#Displaying summary of CNN
summary(model)

#Specify the optimizer, loss function, and metrics
model %>% compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = c('accuracy')
)

#Determining early stopping to minimize computation time when accuracy is not increasing
callback_specs <- list(callback_early_stopping(monitor = 'val_loss', 
                                               min_delta = 0, 
                                               patience = 10,
                                               verbose = 0, 
                                               mode = 'auto'),
                       callback_model_checkpoint(filepath = 'best_model.hdf5', save_freq = 'epoch',
                                                 save_best_only = TRUE))

#Running Optimization
history <- model %>% fit(
  x.train, y.train,
  epochs = 50, 
  batch_size = 128,
  validation_split = 0.2,
  callbacks = callback_specs)

#Loading the best saved model
model_best <- load_model_hdf5('best_model.hdf5', compile = FALSE)

#Computing the model Performance
model %>% evaluate(x.test, y.test)

#Computing predicted values
p_hat_test = model_best %>% predict(x.test)
y_hat_test = apply(p_hat_test,1,which.max)

y_true <- apply(y.test, 1, which.max)
sum(y_hat_test == y_true)/length(y_true)

#Computing value of ROC
library(pROC)
multiclass.roc(y_true, y_hat_test)