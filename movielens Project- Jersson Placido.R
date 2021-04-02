#####################################################################################
# Create the train set (edx) and test set (validation set| final hold-out test set)
####################################################################################


if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
#load the required libraries
library(tidyverse)
library(caret)
library(data.table)
library(lubridate)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                          # title = as.character(title),
                                           #genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

###################################################################
# Create train and test sets from the edx data set
###################################################################
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
edx_test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE) #train set is 80% of the original edx data
edx_train_set<- edx[-edx_test_index,]
edx_test_set <- edx[edx_test_index,]

# Verify the partition
dim(edx_train_set)
dim(edx_test_set)

# Make sure userId and movieId in test set are also in train set
validationedx <- edx_test_set %>% 
  semi_join(edx_train_set, by = "movieId") %>%
  semi_join(edx_train_set, by = "userId")

# Add rows removed from validationedx set back into edx_train_set
removed_edx <- anti_join(edx_test_set, validationedx)
edx_train_set<- rbind(edx_train_set, removed_edx)

# Definition of root mean squared error (RMSE) function

RMSE <- function(predicted_ratings,true_ratings){
  sqrt(mean((predicted_ratings - true_ratings)^2))
}

######################################################################
# Building the Recommendation System utilizing the mean of the rating
######################################################################


med_hat <- mean(edx_train_set$rating) # rating mean calculation

naive_edx_rmse <- RMSE(med_hat, validationedx$rating) # naive rmse calculation

edx_rmse_results <- data_frame(method = "Naive model with just the mean", RMSE = naive_edx_rmse)

edx_rmse_results %>% knitr::kable()

######################################################################
# Recommendation system utilizing movieID
#####################################################################

meanedx <- mean(edx_train_set$rating) #mean calculation
movie_med <- edx_train_set %>% 
  group_by(movieId) %>% 
  summarize(b_im = mean(rating - meanedx)) #mean rating per movieID

# Rating prediction using movieID as unique predictor
predicted_ratings_edx <- meanedx + validationedx %>% 
  left_join(movie_med, by='movieId') %>%
  .$b_im

#Model 1 RMSE CALCULATION
model_1_rmse_1 <- RMSE(predicted_ratings_edx, validationedx$rating)
rmse_results_1 <- bind_rows(edx_rmse_results,
                          data_frame(method="Movie Effect Model",
                                     RMSE = model_1_rmse_1 ))

# models comparison 
rmse_results_1 %>% knitr::kable()

#######################################################
# Recommendation system utilizing movieID and userID
######################################################

user_med <- edx_train_set %>% 
  left_join(movie_med, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_um = mean(rating - meanedx - b_im))

predicted_ratings_2 <- validationedx %>% 
  left_join(movie_med, by='movieId') %>%
  left_join(user_med, by='userId') %>%
  mutate(pred = meanedx + b_im + b_um) %>%
  .$pred

#Model 2 RMSE CALCULATION
model_2_rmse_2 <- RMSE(predicted_ratings_2, validationedx$rating)
rmse_results_2 <- bind_rows(rmse_results_1,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = model_2_rmse_2 ))

# models comparison 
rmse_results_2 %>% knitr::kable()


######################################################################
#Regularized model using movieID
######################################################################

lambda <- 3
movie_reg_med_edx <- edx_train_set %>% 
  group_by(movieId) %>% 
  summarize(b_im = sum(rating - meanedx)/(n()+lambda), n_i = n()) 

predicted_ratings_3 <- validationedx %>% 
  left_join(movie_reg_med_edx, by='movieId') %>%
  mutate(pred = meanedx + b_im) %>%
  .$pred

model_3_rmse_3 <- RMSE(predicted_ratings_3, validationedx$rating)
rmse_results_3 <- bind_rows(rmse_results_2,
                          data_frame(method="Regularized Movie Effect Model",  
                                     RMSE = model_3_rmse_3  ))
rmse_results_3 %>% knitr::kable()

#######################################################################
#Regularized Movie + User Effect Model
######################################################################

lambdas <- seq(0, 10, 0.25)
rmses_edx <- sapply(lambdas, function(l){
  med <- mean(edx_train_set$rating)
  b_im <- edx_train_set %>% 
    group_by(movieId) %>%
    summarize(b_im = sum(rating - med)/(n()+l))
  b_um <- edx_train_set %>% 
    left_join(b_im, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_um = sum(rating - b_im - med)/(n()+l))
  predicted_ratings <- 
    validationedx %>% 
    left_join(b_im, by = "movieId") %>%
    left_join(b_um, by = "userId") %>%
    mutate(pred = med + b_im + b_um) %>%
    .$pred
  return(RMSE(predicted_ratings, validationedx$rating))
})

# lambdas vs RMSE plot
qplot(lambdas, rmses_edx) 

# selected lambda value
lambda_rmses_edx <- data_frame(model= "Regularized Movie + User Effect Model", lambda=lambdas[which.min(rmses_edx)])
lambda_rmses_edx %>% knitr::kable()

#RMSE comparisons
rmse_results_4 <- bind_rows(rmse_results_3,
                          data_frame(method="Regularized Movie + User Effect Model",  
                                     RMSE = min(rmses_edx)))

rmse_results_4 %>% knitr::kable()

#######################################################################
#Regularized Movie + User Effect Model +  Date (week)
######################################################################
library(lubridate)
edx_train_date <- mutate(edx_train_set, date= as_datetime(timestamp)) #transform time stamp in datetime
edx_test_date <- mutate(validationedx, date= as_datetime(timestamp)) #transform time stamp in datetime

lambdas <- seq(0, 10, 0.25)
rmses_edx_date <- sapply(lambdas, function(l){
  med <- mean(edx_train_date$rating)
  b_im <- edx_train_date %>% 
    group_by(movieId) %>%
    summarize(b_im = sum(rating - med)/(n()+l))
  b_um <- edx_train_date %>% 
    left_join(b_im, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_um = sum(rating - b_im - med)/(n()+l))
  d_um <- edx_train_date %>% 
    left_join(b_im, by = "movieId") %>%
    left_join(b_um, by = "userId") %>%
    mutate(date = round_date(date, unit = "week")) %>%
    group_by(date) %>% summarize(d_um = sum(rating - b_im - med- b_um)/(n()+l))
  predicted_ratings12 <- 
    edx_test_date %>% 
    left_join(b_im, by = "movieId") %>%
    left_join(b_um, by = "userId") %>%
    mutate(date = round_date(date, unit = "week")) %>%
    left_join(d_um, by = "date") %>%
    mutate(pred = med + b_im + b_um+ d_um) %>%
    .$pred
  RMSE(predicted_ratings12, edx_test_date$rating)
})

# lambdas vs RMSE plot
qplot(lambdas, rmses_edx_date)  

#selected lambda value
lambda_rmses_edx_date <- data_frame(model= "Regularized Movie + User Effect + Date(week) Model", lambda=lambdas[which.min(rmses_edx_date)])
lambda_rmses_edx_date %>% knitr::kable() #best lambda

#RMSE COMPARISONS
rmse_results_5 <- bind_rows(rmse_results_4,
                            data_frame(method="Regularized Movie + User Effect + Date(week) Model",  
                                       RMSE = min(rmses_edx_date)))

 
rmse_results_5 %>% knitr::kable() 


#######################################################################
#Regularized Movie + User Effect Model + rating date (week)+ genres
######################################################################

library(lubridate)
edx_train_date <- mutate(edx_train_set, date= as_datetime(timestamp)) #transform time stamp in datetime
edx_train_date1 <- mutate(edx_train_date, date = round_date(date, unit = "week"))#transform date in weeks

edx_test_date <- mutate(validationedx, date= as_datetime(timestamp)) #transform time stamp in datetime
edx_test_date1 <- mutate(edx_test_date, date = round_date(date, unit = "week"))#transform date in weeks


options(digits = 5)
lambdas <- seq(0, 10, 0.25)
rmses_edx_date_ge <- sapply(lambdas, function(l){
  med <- mean(edx_train_date1$rating)
  b_im <- edx_train_date1 %>% 
    group_by(movieId) %>%
    summarize(b_im = sum(rating - med)/(n()+l))
  b_um <- edx_train_date1 %>% 
    left_join(b_im, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_um = sum(rating - b_im - med)/(n()+l))
  d_um <- edx_train_date1 %>% 
    left_join(b_im, by = "movieId") %>%
    left_join(b_um, by = "userId") %>%
    group_by(date) %>% summarize(d_um = sum(rating - b_im - med- b_um)/(n()+l))
  g_um <- edx_train_date1 %>%
    left_join(b_im, by = "movieId") %>%
    left_join(b_um, by = "userId") %>%
    left_join(d_um, by = "date") %>%
    group_by(genres)%>% summarize(g_um = sum(rating - b_im - med- b_um - d_um)/(n()+l))
  predicted_ratings14 <- 
    edx_test_date1 %>% 
    left_join(b_im, by = "movieId") %>%
    left_join(b_um, by = "userId") %>%
    left_join(d_um, by = "date") %>%
    left_join(g_um, by = "genres")%>%
    mutate(pred = med + b_im + b_um + d_um + g_um) %>%
    .$pred
  RMSE(predicted_ratings14, edx_test_date1$rating)
})

#PLOT LAMBDAS VS RMSE
qplot(lambdas, rmses_edx_date)  

#SELECTED LAMBDA
lambdae_edx_date_ge <- data_frame(model= "Regularized Movie + User Effect + Date(week) + Genre Model", lambda=lambdas[which.min(rmses_edx_date_ge)])

lambdae_edx_date_ge %>% knitr::kable()

#RSME COMPARISON TABLE
rmse_results_6 <- bind_rows(rmse_results_5,
                            data_frame(method="Regularized Movie + User Effect + Date(week) + Genre Model",  
                                       RMSE = min(rmses_edx_date_ge)))

rmse_results_6 %>% knitr::kable() 


#######################################################################
# VALIDATION OF THE MODEL USING THE ORIGINAL PARTION EDX AND VALIDATION
#Regularized Movie + User Effect Model + rating date (week)+ genres
#######################################################################
library(lubridate)
edx_dates <- mutate(edx, date= as_datetime(timestamp)) #transform time stamp in datetime
edx_dates1 <- mutate(edx_dates, date = round_date(date, unit = "week"))#transform date in weeks

validation_date <- mutate(validation, date= as_datetime(timestamp)) #transform time stamp in datetime
validation_date1 <- mutate(validation_date, date = round_date(date, unit = "week"))#transform date in weeks


options(digits = 5)
lambdas <- seq(0, 10, 0.25)
rmses_edx_final<- sapply(lambdas, function(l){
  medfi <- mean(edx_dates1$rating)
  b_imfi <- edx_dates1%>% 
    group_by(movieId) %>%
    summarize(b_imfi = sum(rating - medfi)/(n()+l))
  b_umfi <- edx_dates1 %>% 
    left_join(b_imfi, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_umfi = sum(rating - b_imfi - medfi)/(n()+l))
  d_umfi <- edx_dates1 %>% 
    left_join(b_imfi, by = "movieId") %>%
    left_join(b_umfi, by = "userId") %>%
    group_by(date) %>% summarize(d_umfi = sum(rating - b_imfi - medfi- b_umfi)/(n()+l))
  g_umfi <- edx_dates1 %>%
    left_join(b_imfi, by = "movieId") %>%
    left_join(b_umfi, by = "userId") %>%
    left_join(d_umfi, by = "date") %>%
    group_by(genres)%>% summarize(g_umfi = sum(rating - b_imfi - medfi- b_umfi - d_umfi)/(n()+l))
  predicted_ratingsf <- 
    validation_date1 %>% 
    left_join(b_imfi, by = "movieId") %>%
    left_join(b_umfi, by = "userId") %>%
    left_join(d_umfi, by = "date") %>%
    left_join(g_umfi, by = "genres")%>%
    mutate(pred = medfi + b_imfi + b_umfi + d_umfi + g_umfi) %>%
    .$pred
  RMSE(predicted_ratingsf, validation_date1$rating)
})

#LAMBDA VS RMSE PLOT
qplot(lambdas, rmses_edx_final)  

#SELECTED LAMBDA
lambda_final<-  data_frame(model= "Regularized Movie + User + Date + Genre, Validation data", lambda=lambdas[which.min(rmses_edx_final)])
lambda_final %>% knitr::kable()

#RMSE OBTAINED
RMSE_Final <- data_frame(model= "Regularized Movie + User + Date + Genre, Validation data", RMSE=min(rmses_edx_final))
RMSE_Final %>% knitr::kable()

#RMSE COMPARISON 
rmse_results_final <- bind_rows(rmse_results_6,
                                data_frame(method="Regularized Movie + User + Date + Genre, Validation data",  
                                           RMSE = min(rmses_edx_final)))

RMSE_Final %>% knitr::kable()
