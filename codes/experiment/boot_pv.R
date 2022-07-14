library(ggplot2)
library(lubridate)
library(quantmod)
library(plotly)
library(moments)
library(PerformanceAnalytics)
library(boot)
library(xtable)

rm(list = ls())

data.file <- "/Users/cheng/Google Drive/PhD/Research/Portfolio Selection via TBN/output/portfolio_return_net.csv"
ds <- read.csv(data.file)
tail(ds)
rownames(ds) <- ds$X
ds$X <- NULL
ds <- as.xts(ds)
# ds <- apply.monthly(ds,apply,2,function(x) prod(x+1) - 1)
ds <- data.frame(Date = date(ds),ds)
rownames(ds) <- NULL
# ds$date <- ceiling_date(ds$Date,"m") - 1

stat_name <- c("Ledoit_linear_identity", "Ledoit_nonlinear_identity", "RL_identity",
               "Ledoit_linear_TBN","RL_TBN")

port_ds <- ds[,c("Date",stat_name)]


FF_file <- "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
temp <- tempfile()
download.file(FF_file,temp)
unz_files <- unzip(temp)
ds <- read.csv(unz_files,skip = 3)
names(ds)[1] <- "date"
ds <- data.frame(apply(ds, 2, as.numeric))
ds$Date <- ymd(ds$date)
ds <- ds[,c("Date","RF")]
ds$RF <- ds$RF/100

ds <- merge(port_ds,ds)
ds[,stat_name]  <- ds[,stat_name] - ds$RF
ds$RF <- NULL

# sr_f <- function(x) sqrt(252)*mean(x)/sd(x)
# apply(ds[,-1],2,sr_f)


ds_xts <- ds
rownames(ds_xts) <- ds$Date
ds_xts$Date <- NULL
ds_xts <- as.xts(ds_xts)
chart.CumReturns(ds_xts[,4:5],legend.loc = "topleft",wealth.index = TRUE)

chart.CumReturns(ds_xts[,1:3],legend.loc = "topleft",wealth.index = TRUE)


round(apply(apply.yearly(ds_xts,apply,2,sr_f),2,mean),3)
############################3
##### let's bootstrap

get.asterisk <- function(pv) {
  star <- rep("",length(pv))
  star[pv < 0.1] <- "*"
  star[pv < 0.05] <- "**"
  star[pv < 0.01] <- "***"
  return(star)
}


my_sum <- function(x) {
  m1 <- mean(x)*252
  s1 <- sd(x)*sqrt(252)
  s2 <- sd(x[x<0])*sqrt(252) # sortino ratio
  sr1 <- m1/s1
  sr2 <- m1/s2
  skew <- skewness(x)
  kurt <- kurtosis(x)
  VaR_1 <- m1 - quantile(x,0.05)
  CVAR <- mean(x[x < quantile(x,0.05)])
  result <- c(Mean = m1*100, Std = s1*100, Std_neg = s2*100, Sharpe = sr1, Sortino = sr2)
  
  return(result)
}  



port_r_boot <- function(port_r_i) { 
  
  boot.f <- function(port_r_i) {
    SUM <- apply((apply.yearly(port_r_i,apply,2,my_sum)),2,mean)
    SUM <- matrix(SUM,5,byrow = FALSE)
    colnames(SUM) <- stat_name
    A2 <- SUM
    A2 <- cbind(A2,A2[,3] - A2[,1],A2[,3] - A2[,2],A2[,5] - A2[,4])
    return( A2[,6:8]  )
  }
  
  set.seed(13)
  boot.metric <- tsboot(port_r_i,statistic =  boot.f,R = 10^3, l = 10, sim = "fixed")
  Z <- t(apply( boot.metric[[2]],1,function(v) v - apply(boot.metric[[2]],2,mean)  ) )
  
  # p.v.A2 <- sapply(1:length(boot.metric[[1]]), function(i)  2*(1-ecdf(Z[,i])(abs(boot.metric[[1]][i] )))    )
  p.v.A2 <- sapply(1:length(boot.metric[[1]]), function(i)  (1-ecdf(Z[,i])((boot.metric[[1]][i] )))    )
  
  p.v.A2 <- get.asterisk(p.v.A2)
  p.v.A2 <- matrix(p.v.A2,5,byrow = FALSE)
  
  SUM <- apply((apply.yearly(port_r_i,apply,2,my_sum)),2,mean)
  SUM <- matrix(SUM,5,byrow = FALSE)
  colnames(SUM) <- stat_name
  A2 <- SUM
  A2 <- cbind(A2,A2[,3] - A2[,1],A2[,3] - A2[,2],A2[,5] - A2[,4])
  
  A3 <- A2
  A3 <- apply(A3,2, function(x) sprintf("%.3f", round(x,3)) )
  A3[,6:8] <- paste(A3[,6:8],p.v.A2,sep = "")
  return(A3)
}


full_sample <- ds_xts
sub_sample <- ds_xts
sub_sample <- sub_sample[!year(ds_xts) %in% c(2001,2007:2009),]




stats_names <- c("Mean","Std","Semi_Std","Sharpe","Sortino")

M1 <- port_r_boot(full_sample)
M2 <- port_r_boot(sub_sample)

rownames(M1) <- rownames(M2) <- stats_names
xtable(M1)


