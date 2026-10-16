args <- commandArgs(trailingOnly=TRUE)
suppressPackageStartupMessages(library(mclust))
x <- as.matrix(read.csv(args[1],header=FALSE))
set.seed(2020)
labels <- integer(0)
selected <- 'FAILED'
for(m in c('EEE','VVV','EII')) {
    result <- tryCatch(Mclust(x,G=as.integer(args[4]),modelNames=m),error=function(e) NULL)
    if(!is.null(result) && !is.null(result$classification)) {
        labels <- result$classification
        selected <- m
        break
    }
}
write(labels,file=args[2],ncolumns=1)
write(selected,file=args[3])
