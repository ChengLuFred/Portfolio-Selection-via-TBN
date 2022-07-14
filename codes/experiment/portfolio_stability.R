library(ggplot2)
library(lubridate)
library(quantmod)
library(plotly)
library(moments)
library(PerformanceAnalytics)
library(boot)
library(xtable)
library(stringr)
library(matrixStats)
library(tidyverse)

rm(list = ls())

# ---
# Data
# ---

file.name.list <- list.files("/Users/cheng/Google Drive/PhD/Research/Portfolio Selection via TBN/output/data/portfolio_weights", pattern="*.csv", full.names=TRUE)
portfolio.weights.list <- lapply(file.name.list, read.csv, row.names = 1)
portfolio.name.list <- sapply(file.name.list, (\(x) str_split(x, '/|.csv')[[1]][11]))
names(portfolio.weights.list) <- portfolio.name.list
strategy.order <- c("Ledoit_linear_identity", 
                    "Ledoit_nonlinear_identity",
                    "RL_identity",
                    "Ledoit_linear_TBN",
                    "RL_TBN")
portfolio.weights.list <- portfolio.weights.list[strategy.order]

# ---
# bootstrap
# ---

get.asterisk <- function(pv) {
  star <- rep("",length(pv))
  star[pv < 0.1] <- "*"
  star[pv < 0.05] <- "**"
  star[pv < 0.01] <- "***"
  return(star)
}

get.HHI <- function(x){
  HHI <- mean(rowSums(x ^ 2))

  return(HHI)
}

get.TO <- function(x) {
  x %>% 
    as.matrix() %>% 
    colDiffs() %>% 
    abs() %>%
    rowSums() %>% 
    mean() -> TO
  
  return(TO)
}


get.stability.with.significance <- function(estimator=get.HHI) {
  # bootstrap
  boot.metric.sum <- lapply(portfolio.weights.list, tsboot, statistic = estimator, R = 10^3, l = 2, sim = "fixed")
  boot.metric.list <- lapply(boot.metric.sum, \(x) x[2])
  boot.metric <- matrix(unlist(boot.metric.list), ncol = length(boot.metric.list))
  boot.metric <- cbind(boot.metric, boot.metric[,3] - boot.metric[,1], boot.metric[,3] - boot.metric[,2], boot.metric[,5] - boot.metric[,4])
  Z <- t(apply( boot.metric,1,function(v) v - apply(boot.metric,2,mean)))
  
  # whole sample estimate
  metric.list <- sapply(boot.metric.sum, \(x) x[1])
  metric.vec <- unlist(metric.list)
  metric.vec <- c(metric.vec, metric.vec[3] - metric.vec[1], metric.vec[3] - metric.vec[2], metric.vec[5] - metric.vec[4])
  
  # p value
  p.v <- sapply(1:dim(Z)[2], function(i) 2 * (1-ecdf(Z[,i])(abs(metric.vec[i]) ) )  )
  p.v.star <- get.asterisk(p.v)
  
  # output
  metric.vec <- round(metric.vec, 3)
  metric.vec[6:8] <- paste(metric.vec[6:8], p.v.star[6:8], sep = "")
  
  return(metric.vec)
}

# ---
# result
# ---

HHI.result <- get.stability.with.significance(get.HHI)
TO.result <- get.stability.with.significance(get.TO)
stability.result <- rbind(HHI.result, TO.result)

xtable(stability.result)


