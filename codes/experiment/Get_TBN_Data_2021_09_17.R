library(qrmdata)
library(quantmod)
library(data.table)
library(igraph)
library(visNetwork)
library(lubridate)
library(plyr)
library(parallel)
library(rvest)

rm(list = ls())
gc()


{
  ### FIND THE S&P 500 CONST. 
  theurl <- "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
  
  file <- read_html(theurl)
  tables <- html_nodes(file, "table")
  
  # try extract each into a data.frame
  tables.list <- numeric()
  for (i in 1:length(tables)) { 
    tables.list[i] <- list(try(html_table(tables[i], fill = TRUE),silent = T))
  }
  tables.list
  # the table we are looking should have columns for volume and price: 
  find.sym <- lapply(tables.list, function(x) grep("Symbol",x,ignore.case = T)   )
  find.sec <- lapply(tables.list, function(x) grep("Security",x,ignore.case = T)   )
  locate.table <- which.max(sapply(find.sym, length) + sapply(find.sec, length) )
  
  ds <- tables.list[[locate.table]][[1]]
  ds$Symbol
  
  #### LOAD CRSP DATA ######
  getwd()
  file.i <- "CRSP_1960_2019_d.csv" # CRSP data?
  
  select_var <- c("PERMCO","PERMNO","date",
                  "COMNAM","SHROUT","RET","PRC",
                  "TICKER","CUSIP","SHRCD")
  DS <- fread(file.i,select = select_var)
  # keep tickers that show up in the wiki page for being part of S&P
  DS <- DS[DS$TICKER %in% ds$Symbol,]
  gc()
  
  
  DS <- DS[DS$SHRCD %in% 10:11,]
  DS <- DS[!is.na(DS$PRC),]
  DS <- DS[!DS$PRC %in% (-(4:9)*11),]
  
  DS$RET <- as.numeric(DS$RET)
  DS$date <- ymd(DS$date)
  
  # let's pivot the data
  DS <- unique(DS)
  
  DS_N <- DS[,.N, by = c("date","TICKER")]
  table(DS_N$N)

  DS[DS$date == 20091215 & DS$TICKER == "CMG",]
      
  
  
  DS$PRC <- abs(DS$PRC)
  DS$MKTCAP <- DS$PRC*DS$SHROUT
  
  
  DS_sub <- unique(DS[,list(PERMNO,date,RET,MKTCAP),])
  DS_sub <- DS_sub[order(DS_sub$PERMNO,DS_sub$date),]
  DS_sub <- na.omit(DS_sub)
  DS_sub[,N :=  .N, by = c("date","PERMNO")]
  table(DS_sub$N)
  DS_sub$N <- NULL
  
  uniqe_permnos <- unique(DS_sub$PERMNO)
  set.seed(13)
  sample_permnp <- sample(uniqe_permnos,3)
  DS_sub_sample <- DS_sub[DS_sub$PERMNO %in% sample_permnp,]
  
  DS_cast <- dcast.data.table(DS_sub,date~PERMNO, value.var = "RET" )
  
  
  # keep data from the 90 and forward
  DS_cast <- DS_cast[DS_cast$date >= "1990-01-01",] 
  
  # keep stocks with full data 
  na_count <- DS_cast[,lapply(.SD,function(x) sum(is.na(x)) ),]
  DS_cast <- DS_cast[,na_count == 0,with = F]
}


####### MERGE WITH COMPUSTAT to find gvkey


file.j <- "/home/simaan/Dropbox/Data/DATA/COMPUSTAT_1960_2020.csv"
select.var2 <- tolower(c("cusip","gvkey","datadate"))
DT2  <- unique(fread(file.j,select = select.var2))
DT2 <- unique(DT2)
DT2$CUSIP <- substr(DT2$cusip,0,8)
DT2$cusip <- NULL

DT2 <- DT2[DT2$CUSIP %in% DS$CUSIP,]
DT2$date <- ymd(DT2$datadate)
DT2$datadate <- NULL
DT2$date  <- floor_date(DT2$date,"m") + month(6)
DT2$date <- ceiling_date(DT2$date,"m") - 1


DT <- DS[,list(PERMNO,date,TICKER,CUSIP)]
DT$date <- ceiling_date(DT$date,"q") - 1
DT <- unique(DT)

DT12 <- merge(DT,DT2,by = c("CUSIP","date"))
DT12 <- DT12[order(DT12$TICKER),]
DT12$CUSIP <- NULL
DT12 <- unique(DT12)

# save the index data
file.i <- "/home/simaan/Dropbox/Stevens/Students/PhD/Cheng Lu/Data/gvkey_id.csv"
write.csv(DT12,file.i)

# finally save the return data
DS_cast <- DS_cast[,names(DS_cast) %in% c("date",DT12$PERMNO),with = F]

file.i <- "/home/simaan/Dropbox/Stevens/Students/PhD/Cheng Lu/Data/permno_ret.csv"
write.csv(DS_cast,file.i)

