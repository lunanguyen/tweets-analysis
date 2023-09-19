library(datasets)  # Load base packages manually
library(sjPlot)
# Installs pacman ("package manager") if needed
if (!require("pacman")) install.packages("pacman")

# Use pacman to load add-on packages as desired
pacman::p_load(pacman, caret, lars, tidyverse, rio, ggfortify, ggplot2, dplyr, broom)
install.packages('lmtest', dependencies = TRUE)
library(lmtest)
library(dynlm)
library(rio)
library(sjPlot)
library(tidyverse)
# LOAD DATA ################################################

# Import csv file that has tweets features summary and stock prices
data1 <- import("C:\\Users\\nguye\\Twitter\\Linear Regression\\Period 1\\Jan_Period 1.csv", header=TRUE)
data2 <- import("C:\\Users\\nguye\\Twitter\\Linear Regression\\Period 2\\Feb_Period 2.csv", header=TRUE)
data3 <- import("C:\\Users\\nguye\\Twitter\\Linear Regression\\Period 3\\Mar_Period 3.csv", header=TRUE)

# Import csv file that has sentiment data summary and stock prices
sentiment1 <- import("C:\\Users\\nguye\\Twitter\\Linear Regression\\Period 1\\Correlation_Sentiment1.csv", header=TRUE)
sentiment2 <- import("C:\\Users\\nguye\\Twitter\\Linear Regression\\Period 2\\Correlation_Sentiment2.csv", header=TRUE)
sentiment3 <- import("C:\\Users\\nguye\\Twitter\\Linear Regression\\Period 3\\Correlation_Sentiment3.csv", header=TRUE)


# Print out the correaltion value for sentiment data vs. stock prices
tab_corr(sentiment1[-1], corr.method = "pearson", p.numeric = TRUE)
tab_corr(sentiment2[-1], corr.method = "pearson", p.numeric = TRUE)
tab_corr(sentiment3[-1], corr.method = "pearson", p.numeric = TRUE)

# Create a model and uses the model to create a best fit line.
# Print out the correlation value for tweets features vs. stock prices.
# Print our summary
regression3 <- lm(Cov_Likes ~ GOOG, data = data3)
summary(regression3)

regression3 <- lm(Cov_Likes ~ MSFT, data = data3)
summary(regression3)

regression3 <- lm(Cov_Likes ~ AAPL, data = data3)
summary(regression3)

regression3 <- lm(Cov_Likes ~ AAL, data = data3)
summary(regression3)

regression3 <- lm(Cov_Likes ~ MAR, data = data3)
summary(regression3)

regression3 <- lm(Cov_Likes ~ IT, data = data3)
summary(regression3)

regression3 <- lm(Cov_Likes ~ Hotel, data = data3)
summary(regression3)

regression3 <- lm(Cov_Likes ~ Airlines, data = data3)
summary(regression3)

regression3 <- lm(Cov_Likes ~ SP500, data = data3)
summary(regression3)

regression3 <- lm(Cov_Likes ~ Russel2000, data = data3)
summary(regression3)

# Get critical F value
qf(0.95, 1, 8)



