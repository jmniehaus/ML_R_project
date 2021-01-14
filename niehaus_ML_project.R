#====================================================
# JOHN NIEHAUS 
# UIN 225009469
# STAT639 PROJECT
# ALL PARTS (CLASSIFICATION AND UNSUPERVISED)
# SPRING 2020 --- The Year of the Corona Virus
# 
#====================================================

# 1. Set up. Get packages, etc. 
rm(list=ls())
gc()
#setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

options(Ncpus=parallel::detectCores()) # used to install packages in parallel

pks = c("glmnet", 
        "e1071", 
        "foreach", 
        "randomForest", 
        "doParallel", 
        "class", 
        "MASS",
        "xgboost", 
        "mclust", 
        "dbscan",
        "xtable")

needed = setdiff(pks, rownames(installed.packages()))
if(length(needed) > 0 ) install.packages(needed, repos="https://cloud.r-project.org")
sapply(pks, library, character.only=T)

today = format(Sys.time(), "%Y_%m_%d")

sink(paste0("benchmark_", today, ".txt"))
RNGkind("L'Ecuyer-CMRG") # for parallel reproducibility
set.seed(081360)

#  2. Preliminary functions, setup data
load("class_data.RData")
Xmat = as.matrix(x)
p = ncol(Xmat)
n = nrow(Xmat)
Xstd = scale(Xmat)
Xdf = x
ncores = parallel::detectCores()

make.folds = function(data, nfolds){
  if(nfolds %% 1 != 0 || length(nfolds) != 1) stop("Arg `nfolds` must be integer scalar.")
  if(is.null(nrow(data))) stop("Arg `data` must be a dataframe or matrix.")
  
  ids = seq_len(nfolds)
  ids = sample(ids, size=nrow(data), replace=T)
  return(ids)
}


mis.class = function(pred.class, true.class, print.table=FALSE){
  if(length(pred.class) != length(true.class)) stop("`pred.class` and `true.class` not conformable.")
  
  confusion.mat = table(pred.class, true.class)
  rate = 1-sum(diag(confusion.mat))/sum(confusion.mat)
  
  if(!print.table)
    return(rate) 
  else
    return(list(rate = rate, confusion.mat = confusion.mat))
  
}

normalize = function(data){
  return(apply(data, 2 , function(x) (x-min(x))/(max(x) - min(x))))
}

#============================================
# Setup Grids and folds for cross validation 
#============================================

# Inner and outer folds for nested CV
# NOTE: Numbers are fold IDs. List indices for inner_folds show which outer fold was left out. 
# Example: inner_folds[[3]] is the vector of fold IDs for the inner CV loop when 
# outer_fold 3 is left out. There are `ninner` unique inner fold IDs, `nouter` unique outer fold IDs.

nouter = 6
ninner = 5
cv_rep = 5

outer_folds = list()
inner_folds = list()

for(i in seq_len(cv_rep)){
  outer_folds[[i]] = make.folds(Xmat, nfolds = nouter)
  inner_folds[[i]] = lapply(seq_len(nouter), function(x) make.folds(Xmat[outer_folds[[i]] != x,], nfolds = ninner))
}

### Grids for each method where applicable
## Elastic Net grid
alpha = seq(0,1,by=.1)

# find the maximum value of lambda such that all coefs are zero using http://www.jstatsoft.org/v33/i01/paper
find.max.lambda = function(X, y, alpha, size=NULL, standardize=T){
  
  alpha = ifelse(alpha == 0, .001, alpha)
  
  if(!is.null(size)) {
    X = X[sample(nrow(X), size=size, replace=F),]
    y = y[sample(nrow(X), size=size, replace=F)]
  }
  
  if(standardize) X=scale(X)
  
  abs_inner = abs(crossprod(y,X))
  max_abs = max(abs_inner)
  lam_max = max_abs /(nrow(X)*alpha) 
  
  return(lam_max)
}


# Generate sequence of lambda values on logarithmic scale up to log(lam_max) from previous step
gen_lam_grid = function(lam_max, data, alpha=NULL, length.out){
  if(nrow(data) < ncol(data)) delta = .01 else delta = .001
  if(is.null(alpha)) alpha = 1
  
  sort( exp( seq(log(delta*lam_max), log(lam_max), length.out = length.out) ), decreasing=T)
  
} 



size = sum(inner_folds[[1]][[1]] != 1)
samples = rep(size, 5000)

#Randomly permute data to find single acceptable range of lambdas for all possible training sets
lam_bound = mclapply(samples, 
                     function(x) find.max.lambda(Xmat, y, alpha, size=x),
                     mc.cores=ncores, 
                     mc.preschedule=T)

lam_bound = do.call(rbind, lam_bound)
lam_max_a = apply(lam_bound, 2, max)

lam_grid = lapply(lam_max_a, gen_lam_grid, data = Xmat[1:size,], length.out=100, alpha=alpha)
names(lam_grid) = alpha




## Random forest grid
mtry.grid = c(floor(sqrt(p)), floor(sqrt(p)) + 50, seq(100, 500, 50))

## KNN grid
knn.grid = seq(1, 50)

## Boosting grid
boost.grid = expand.grid(ntrees = 500,
                         learning = c(.0001,.001, .01, .1, .3), 
                         depth=c(1,3,5,7,9),
                         colsamp = c(sqrt(p)/n, .25, .5, 1),
                         gamma= c(0, 0.01, 0.1, 0.5, 1)
)

## SVM Grid
svm.cost = c(.01, seq(.05, 2, length.out = 20), 10, 100, 1000, 10000, 100000)
svm.radial = expand.grid(gamma = c( 10^(-5:0), .5, 2:5), 
                         cost = svm.cost,
                         kernel = "radial",
                         degree = 1,
                         stringsAsFactors = F)

svm.linear = expand.grid(gamma = 1, 
                         cost = svm.cost,
                         kernel = "linear",
                         degree = 1,
                         stringsAsFactors = F)

svm.poly = expand.grid(gamma = 1, 
                       cost = svm.cost,
                       kernel = "polynomial",
                       degree = c(2:8),
                       stringsAsFactors = F)

svm.grid = rbind.data.frame(svm.linear, svm.poly, svm.radial)

#==============================================================================================
#Cross validation 
#==============================================================================================
#setup parallel backend
ncores = parallel::detectCores()
cl = makeCluster(ncores)
registerDoParallel(cl)
#registerDoRNG()



seeds = sample(seq(111111, 999999), 30)
seeds = split(seeds, ceiling(seq_along(seeds)/nouter))
start = Sys.time()
# Repeat, nested cross validation over grid
results = # 
  foreach(cvrep = seq_len(cv_rep)) %:%
  foreach(f = seq_len(nouter),
          .packages= c("glmnet", "randomForest", "class", "e1071", "xgboost"),
          .errorhandling = "pass") %dopar%  {
            
            set.seed(seeds[[ cvrep ]][[ f ]])
            
            #Initialize data and storage containers
            #Recall: inner_folds[[f]] is the vector of inner fold ids when outer_fold f is left out
            outer.id = outer_folds[[ cvrep ]]
            inner.id = inner_folds[[ cvrep ]][[ f ]]
            
            X.inner = Xmat[outer.id != f, ]
            y.inner = as.factor(y[outer.id != f ])
            y.inner.num = as.numeric(as.character(y.inner))
            
            rfs = vector(mode="list", length = length(mtry.grid))
            names(rfs) = paste0("mtry=", mtry.grid)
            
            knns.std = knns = vector(mode="list", length=length(knn.grid))
            names(knns) = paste0("K=", knn.grid)
            
            nbayes =  vector(length=ninner)
            names(nbayes) = paste0("inner.fold=", seq_len(ninner))
            
            boosts = vector(mode = "list", length = nrow(boost.grid) )
            
            inner.glms = vector(mode="list", length=length(alpha))
            
            svms = vector(mode="list", length = nrow(svm.grid))
            
            
            # Start logistic Elastic Net inner cv
            iter = 0
            for(a in alpha){
              iter = iter + 1
              inner.glms[[iter]] = cv.glmnet(X.inner,
                                             y.inner,
                                             lambda = lam_grid[[ paste(a) ]],
                                             alpha = a,
                                             foldid = inner.id,
                                             family="binomial",
                                             type.measure = "class")
            }
            
            lambdas = lapply(inner.glms, function(x) x$lambda)
            glm.cv = lapply(inner.glms, function(x) x$cvm)
            glm.cv = mapply(cbind.data.frame, lambda=lambdas, alpha=alpha, cv=glm.cv, SIMPLIFY=F)
            glm.cv = do.call(rbind.data.frame, glm.cv)
            
            
            # inner cv for boosting
            xg.folds = lapply(sort(unique(inner.id)), function(x) which(inner.id==x)) #validation fold index
            
            for(b in seq_len(nrow(boost.grid))){
              if(b == 1) message("Starting boosting iterations for thread", f, " and cv repitition ", cvrep, " at ", Sys.time())
              if(b %% (nrow(boost.grid)/10) == 0) message(b/nrow(boost.grid)*100, "% complete for thread ", f, ", cvrep ", cvrep, " at ", Sys.time())
              
              boosts[[b]]=xgb.cv(data = X.inner, 
                                 label = y.inner.num, 
                                 colsample_bynode = boost.grid[ b, "colsamp" ], 
                                 eta = boost.grid[ b, "learning" ],
                                 gamma = boost.grid[b, "gamma"],
                                 max_depth = boost.grid[ b, "depth" ], 
                                 objective = "binary:logistic",
                                 nrounds = boost.grid[ b, "ntrees" ],
                                 verbose = 0, 
                                 nthread = 1,
                                 folds = xg.folds,
                                 early_stopping_rounds = 40,
                                 prediction = T)
              
            }
            
            tree.opt = sapply(boosts, function(x) x$best_ntreelimit )
            boost.cv = lapply(boosts, function(x) x$evaluation_log[x$best_ntreelimit,])
            boost.cv = sapply(boost.cv, function(x) x$test_error_mean )
            boost.cv = cbind.data.frame(boost.grid, tree.opt=tree.opt, cv=boost.cv)
            
            #Loop over inner folds when outer_fold[[f]] is left out
            for(i in seq_len(ninner)){
              
              X.tr = X.inner[ inner.id != i, ]
              X.tr.std = scale(X.tr)
              X.tr.df  = as.data.frame(X.tr)
              
              X.test = X.inner[inner.id == i, ]
              X.test.std = scale(X.test)
              X.test.df  = as.data.frame(X.test)
              
              y.tr = y.inner[inner.id != i]
              y.tr.num = as.numeric(as.character(y.tr))
              
              y.test = y.inner[inner.id == i]
              y.test.num = as.numeric(as.character(y.test))
              
              
              #Random Forest
              
              parm.iter = 0
              for(m in mtry.grid){
                
                parm.iter = parm.iter + 1
                message("Starting RF iteration ", i , " for thread ", f, " at ", Sys.time())
                
                rf = randomForest(X.tr,
                                  y.tr,
                                  mtry = m,
                                  ntree = 5000,
                                  xtest = X.test,
                                  ytest = y.test,
                                  proximity = F)
                
                rfs[[ paste0("mtry=", m) ]][[ i ]] = mis.class(rf$test[["predicted"]], y.test)
                
              } # end RF loop
              
              # Start KNN
              parm.iter = 0
              for(k in knn.grid){
                parm.iter = parm.iter + 1
                
                knns.std[[ parm.iter ]][[ i ]] = mis.class(knn(train = X.tr.std,
                                                               test  = X.test.std,
                                                               cl    = y.tr,
                                                               k     = k),
                                                           y.test)
                
                knns[[ parm.iter ]][[ i ]] = mis.class(knn(train = X.tr,
                                                           test  = X.test,
                                                           cl    = y.tr,
                                                           k     = k),
                                                       y.test)
                
              } # end knn loop
              
              
              
              # Start Naive Bayes
              nb.obj = naiveBayes(x=X.tr, y=y.tr)
              nb.pred = predict(nb.obj, X.test)
              nbayes[i] = mis.class(nb.pred, y.test)
              
              
              #Start SVM loop
              
              for(s in seq_len(nrow(svm.grid))){
                
                svm.mod = svm(x=X.tr.std, 
                              y=y.tr, 
                              scale=F, 
                              type="C", 
                              kernel = svm.grid[s, "kernel"],
                              degree = svm.grid[s, "degree"],
                              cost = svm.grid[s, "cost"],
                              gamma= svm.grid[s, "gamma"],
                              coef0 = 1)
                
                svm.preds = predict(svm.mod, X.test.std)
                
                svm.mclass = mis.class(svm.preds, y.test)
                
                svms[[s]][[i]] = svm.mclass 
              }
              
              
              
              
              
              
              
            } # end inner folds loop
            
            #get inner cv for all but glm and boost (see previous code for those two means)
            rf.cv = sapply(rfs, mean)
            rf.cv = cbind.data.frame(mtry = mtry.grid, cv=rf.cv)
            
            knn.cv = sapply(knns, mean)
            knn.std.cv = sapply(knns.std, mean)
            knn.cv = cbind.data.frame(knn.grid, knn.cv = knn.cv)
            knn.std.cv = cbind.data.frame(knn.grid, knn.std.cv)
            
            svm.cv = sapply(svms, mean)
            svm.cv = cbind.data.frame(svm.grid, svm.cv)
            
            nbayes.cv = mean(nbayes)
            
            return(list(glm.cv=glm.cv,
                        boost.cv=boost.cv,
                        rf.cv = rf.cv, 
                        knn.cv = knn.cv,
                        knn.std.cv=knn.std.cv,
                        svm.cv = svm.cv,
                        nbayes.cv = nbayes.cv)
            )
          } # end foreach


Sys.time() - start
stopCluster(cl)

session = sessionInfo()
save.image(file=paste0("supervised_", today, ".RData"))


extract_min = function(outer.cv.list){
  
  outer_mins = vector(mode = "list", length = length(outer.cv.list))
  names(outer_mins) = paste0("outer_leftout_", seq_along(outer.cv.list), "_min")
  
  for(f in seq_along(outer.cv.list)){
    data = outer.cv.list[[ f ]]
    mins = lapply(data, function(x) if(is.list(x)) x[ which.min(x[, grepl("cv", names(x))]),] else x)
    outer_mins[[f]] = mins[[which.min(sapply(mins, function(x) if(is.list(x)) x[,grepl("cv", names(x))] else x))]]
    
  }
  
  return(outer_mins)
}

#get CV min parms from inner loops to estimate test error
cv.mins.nested = lapply(results, extract_min)


#estimate test error using min parms from inner cv
test.est = vector()
for(rep in seq_len(cv_rep)){
  outer.est = vector()
  for(outer in seq_len(nouter)){
    
    outer.id = outer_folds[[rep]]
    
    X.tr = Xmat[outer.id != outer,]
    X.test = Xmat[outer.id == outer,]
    
    y.tr = y[outer.id != outer]
    y.test = as.factor(y[outer.id == outer])
    
    min.parms = cv.mins.nested[[rep]][[outer]]
    
    
    mod = xgboost(data = X.tr,
                  label = y.tr,
                  colsample_bynode = min.parms[1, "colsamp"],
                  eta = min.parms[1, "learning"],
                  gamma = min.parms[1, "gamma"],
                  max_depth = min.parms[1, "depth"],
                  nrounds = min.parms[1, "tree.opt"],
                  objective = "binary:logistic",
                  verbose = 0,
                  nthread = 1)
    
    pred = ifelse(predict(mod, X.test) >.5 , 1, 0)
    mclass = mis.class(pred, y.test)
    
    outer.est[outer] = mclass
    
  }
  
  test.est[rep] = mean(outer.est)
}

#test error here
test_error = mean(test.est)




### Find optimal parameters now that we have estimate of test error
cv_rep = 5
nfolds = 5
select.folds = lapply(rep(nfolds,cv_rep), make.folds, data=Xmat)

seeds = sample(seq(111111, 999999), cv_rep*nrow(boost.grid), replace=F)
seeds = split(seeds, ceiling(seq_along(seeds)/nrow(boost.grid)))

boost.grid$ntrees = 250

#setup parallel backend
ncores = parallel::detectCores()
cl = makeCluster(ncores)
registerDoParallel(cl)


models =
  foreach(cvrep = seq_len(cv_rep)) %:%
  foreach(b = seq_len(nrow(boost.grid)),
          .packages = "xgboost",
          .errorhandling="pass") %dopar% {
            
            set.seed(seeds[[cvrep]][[b]])
            
            fold.id = select.folds[[cvrep]]
            
            boost.mclass = list()
            
            for(f in sort(unique(fold.id))){
              
              X.tr = Xmat[fold.id != f, ]
              y.tr = y[fold.id != f ]
              
              X.test = Xmat[fold.id == f, ]
              y.test = y[fold.id == f]
              
              
              mod = xgboost(data = X.tr,
                            label = y.tr,
                            colsample_bynode = boost.grid[ b, "colsamp" ],
                            eta = boost.grid[ b, "learning" ],
                            gamma = boost.grid[b, "gamma"],
                            max_depth = boost.grid[ b, "depth" ],
                            objective = "binary:logistic",
                            nrounds = boost.grid[ b, "ntrees" ],
                            verbose = 0,
                            nthread = 1
              )
              
              probs = lapply(seq(1,250,1), function(x) predict(mod, X.test, ntreelimit = x))
              preds = lapply(probs, function(x) ifelse(x < .5, 0, 1))
              mclass = sapply(preds, mis.class, true.class = y.test)
              
              boost.mclass[[f]] = mclass
              
            }
            
            boost.mclass = do.call(rbind, boost.mclass)
            boost.mclass.cv = apply(boost.mclass, 2, mean)
            
            
            
            return(boost.mclass.cv)
          }

stopCluster(cl)


models.comb = lapply(models, function(x) do.call(rbind, x))
names(models.comb) = paste0("cvmat", seq_len(cv_rep))
list2env(models.comb, globalenv())
models.comb = array(c(cvmat1, cvmat2, cvmat3, cvmat4, cvmat5), dim=c(nrow(boost.grid), 
                                                                     boost.grid$ntrees[1],
                                                                     cv_rep)
)
models.cv = apply(models.comb, c(1,2), mean)
models.cv.min.ind = which(models.cv == min(models.cv), arr.ind = T)
fin.mod.parms = boost.grid[models.cv.min.ind[1],]
fin.mod.parms$ntrees = models.cv.min.ind[2]
models.cv[models.cv.min.ind[1], models.cv.min.ind[2]]


# Boosting table
nested.table = unlist(cv.mins.nested, recursive=F)
nested.table = do.call(rbind.data.frame, nested.table)
nested.table$outer = rep(seq(1,6), times=5)
nested.table$repitition = rep(1:5,each=6)
rownames(nested.table) = NULL
xtable(nested.table[1:6,-c(1,8:9)], digits = 4)

final.mod = xgboost(data=Xmat,
                    label=y,
                    colsample_bynode = fin.mod.parms[ 1, "colsamp" ],
                    eta = fin.mod.parms[ 1, "learning" ],
                    gamma = fin.mod.parms[1, "gamma"],
                    max_depth = fin.mod.parms[ 1, "depth" ],
                    objective = "binary:logistic",
                    nrounds = fin.mod.parms[ 1, "ntrees" ],
                    verbose= 0)

final.pred = predict(final.mod, as.matrix(xnew))
ynew = ifelse(final.pred < .5, 0, 1)


save(test_error, ynew, file="225009469.RData")

save.image(paste0("supervised_all_", today, ".RData"))

set.seed(100498)
# Boosting Plot
inds = sample(nrow(boost.grid), size=29, replace=F)
pdf("boosting_paths.pdf")
plot(1:250, models.cv[models.cv.min.ind[1],],
     type="l", col=2, lwd=3.5,
     xlab="N Trees", ylab="CV Error", main="Repeated CV Error for Optimal and Random Boosting Parameters")
for(i in seq_along(inds)){
  lines(models.cv[ inds[i], ], lty=1)
}
abline(v=models.cv.min.ind[2], lty=2, lwd=3, col=2)
dev.off()



#==================================================================================
#********************************************************************
#==================================================================================
#==================================================================================
#              start unsupervised learning ========================================
#==================================================================================
#==================================================================================
#********************************************************************
#==================================================================================


set.seed(122446)


rm(list=ls())
load("cluster_data.RData")
dim(y)

n = nrow(y)
p = ncol(y)

y.std = scale(y)

# Check NORMALITY assumptions for KMEANS and GMM 
### Check multivariate normality using beta distribution of mahalanobis distances
alp = (p-2)/(2*p)
bet = (n-p-3)/(2*(n-p-1))
a   = p/2
b   = (n-p-1)/2

pr = (1:n - alp)/(n - alp - bet + 1)
quantiles = qbeta(pr, a, b)

y.std = scale(y, center=T, scale = F)
S = cov(y.std)
S_inv = solve(S)
dsq = diag(y.std%*% tcrossprod(S_inv, y.std))

u = (n *dsq) / (n-1)^2

pdf("beta_quant.pdf")
par(pin=c(2.75, 2.75))
plot(quantiles,sort(u),type='l',xlab='beta quantile',ylab = 'u quantile', main="Checking Multivariate Normality")
dev.off()


# Marginal normality check
marginals = y.std[,sample(ncol(y), 25, replace=F)]

par(mfrow= c(5,5), pin=c(.75,.75))
check.norm = function(variable){
  qqnorm(variable, main="")
  qqline(variable, main="")
}


apply(marginals, 2, check.norm)
dev.copy(pdf, "marginals.pdf")
dev.off()




  ###Start CLustering here
# PCA to get lower dimensions for clustering 

svd.y = svd(y.std)

u = svd.y$u
d = svd.y$d
v = svd.y$v
PCs = u%*%diag(d)

eigs = d**2/(n-1)
varprop = cumsum(eigs)/sum(eigs)


# Get Subsets of PCs
pc.ind=c(5,20,40)
PCs.sub = lapply(pc.ind, function(x) PCs[,1:x])
names(PCs.sub) = paste0("y.pc.", pc.ind)


# Proportion of variance and scree plots
pdf("pve_scree.pdf")
nvars = c(p, 100, 40)
par(mfrow = c(2,3), pin=c(1.25,1.25),mar=c(3.5,3.5,2.5,1))
for(i in nvars){
  if(i==head(nvars,1)){
    plot(seq_len(i), varprop[1:i], type = "l", xlab="", ylab="")
    title(ylab="Prop of Variance Explained", cex.lab=1.2, line=2)
    abline(v=pc.ind[3], col=4)
  } else if(i==nvars[2]) {
    plot(seq_len(i), varprop[1:i], type = "p", xlab="", ylab="")
    abline(v=pc.ind, col=2:4)
  } else{
    plot(seq_len(i), varprop[1:i], type = "p", xlab="", ylab="")
  }
}


for(i in nvars){
  if(i==nvars[1]){
    plot(seq_len(i), eigs[1:i], type="l", xlab="", ylab="")
    title(ylab="Eigenvalue", cex.lab=1.2, line=2)
    abline(v=pc.ind[3], col=4)
  } else if(i==nvars[2]){
    plot(seq_len(i), eigs[1:i], type="p", xlab="", ylab="")
    abline(v=pc.ind, col=2:4)
  } else if(i==nvars[3]){
    plot(seq_len(i), eigs[1:i], type="p", xlab="", ylab="")
  }
}
title(main= "Scree and Proportion of Variance Plots for PCA", outer=T, line = -2)
title(xlab = "Principal Component Number", outer=T, line=-1, cex.lab=1.4)
dev.off()





#kmeans # use Calinski and Harabasz (1974) ch index
ch.index = function(x,kmax,iter.max=100,nstart=25,
                    algorithm="Lloyd")
{
  ch = numeric(length=kmax-1)
  n = nrow(x)
  for (k in 2:kmax) {
    a = kmeans(x,k,iter.max=iter.max,nstart=nstart,
               algorithm=algorithm)
    w = a$tot.withinss
    b = a$betweenss
    ch[k-1] = (b/(k-1))/(w/(n-k))
  }
  return(list(k=2:kmax,ch=ch))
}

ch = lapply(PCs.sub, ch.index, kmax=25)

pdf("kmeans_ch.pdf")
par(mfrow=c(1,3))
for(i in 1:3){
  ch.i = ch[[i]]
  plot(2:25, ch.i$ch, xlab="", ylab="", cex=1.1)
  
  if(i==1) title(ylab="CH Index", line=2, cex.lab=1.2)
  
  title(xlab = "K", line=2, cex.lab=1.2)
  title(main=paste(pc.ind[i], "PCs Kept"), line=1)
  abline(v=ch.i$k[which.max(ch.i$ch)], col=2)
}
title(main = "CH Index Across Various K in K-Means Clustering", outer=T, line=-1)
dev.off()



#Gaussian Mixture Models

BICs = lapply(PCs.sub, mclustBIC)

pdf("GMM_BIC.pdf")
par(mfrow=c(1,3), pin=c(1.5,3.5))
for(i in 1:3){
  plot(BICs[[i]], xlab="Number of Clusters", main=paste(pc.ind[i], "PCs Kept"), ylab="")
}
title(main="Bayesian Information Criteria Across Number of Clusters for GMM", outer=T, line=-1)
dev.off()



### Hierarchical Clustering
# Function to get within and between cluster sum of squares from hclust
clust.SS = function(data, cluster){

  twss = vector()
  tss = vector()
  bss = vector()
  wss = list()
  cluster = as.matrix(cluster)
  
  for(j in seq_len(ncol(cluster))){
    ss = aggregate(data, 
                   by=list(cluster[,j]), 
                   function(x) sum(scale(x, scale=F)**2))
    wss[[j]] = rowSums(ss[,-1])
    twss[j] = sum(ss[,-1])
    tss[j] = sum(scale(data, scale=F)**2)
    bss[j] = tss[j] - twss[j]
  }
  
  ss.all = list(tss=tss, wss=wss, twss=twss, bss=bss)
    
  return(ss.all) 
      
}

#function for computing CH index
ch.index2 = function(data, kmax, twss, bss){
  
  ch = numeric(length=kmax-1)
  n = nrow(data)
  for (k in 2:kmax) {
    to = twss[k-1]
    b = bss[k-1]
    ch[k-1] = (b/(k-1))/(to/(n-k))
  }
  return(list(k=2:kmax,ch=ch))
}

# Get distances for all subsets of the PC space considered
dists = lapply(PCs.sub, dist)

hc.complete = lapply(dists, hclust, method="complete")
hc.single   = lapply(dists, hclust, method="single")
hc.average  = lapply(dists, hclust, method="average")

clusts = list(hc.complete, hc.single, hc.average)

# Function to plot CH and Total WSS for hclust across linkages, pc components kept, and values of K
plot.ch.wss = function(ch.list, wss.list, linkage){
  par(mfrow=c(2,3), mai=c(.4,.5,.4,.2))
  for(i in 1:3){
    data = ch.list[[i]]
    plot(data$k, data$ch, ylab="", xlab="", xaxt="n")
    axis(1, labels=F)
    abline(v = which.max(data$ch)+1, col=2)
    if(i==1) title(ylab="CH Index", line=2, font.lab=2, cex.lab=1.2)
    title(main=paste(pc.ind[i], "Components Kept"), line = .5)
  }
  for(i in 1:3){
    data = wss.list[[i]]
    plot(2:25, data$twss, ylab="", xlab="")
    title(xlab = "K", font.lab=2, line = 2)
    if(i==1) title(ylab="Total WSS", line=2, cex.lab=1.2, font.lab=2)
  }
  title(paste("CH Index and Total WSS For", linkage,  "Linkage Clustering Across PC Spaces"), outer=T, line=-1)
}



methods = c("Complete", "Single", "Average")

for(method in seq_along(clusts)){
  cuts = lapply(clusts[[method]], function(x) cutree(x,k=2:25))
  ss = mapply(clust.SS, cluster=cuts, data=PCs.sub, SIMPLIFY = F)
  ch = lapply(ss, function(x) ch.index2(data=y, kmax=25, twss=x$twss, bss=x$bss))
  pdf(paste0("ch_", methods[method], ".pdf"))
  plot.ch.wss(ch, ss, linkage=methods[method])
}




# Density Based clustering
#https://iopscience.iop.org/article/10.1088/1755-1315/31/1/012012/pdf

#eps.crit = list(4.9, c(4.9, 12), c(4.9, 19))
# plot.dbscan.eps = function(data.list, n.neig){
#   par(mfrow=c(1,3))
#   for(d in seq_along(data.list)){
#     data = data.list[[d]]
#     distance = as.matrix(dist(data))
#     nn = apply(distance, 2, function(x) mean(sort(x[x != 0])[1:n.neig], na.rm=T))
#     plot(1:nrow(data), sort(nn), type="l", xlab="", ylab="" )
#     title(xlab="Obs Number", ylab=paste("Distance to", n.neig, "Nearest Neighbors"), line=2)
#     title( main=paste(pc.ind[d], "Components Kept"), line=.5)
#     title(main = "Distance to Nearest Neighbor Across PC Spaces", outer=T, line=-1)
#     abline(h=eps.crit[[d]], lty=2)
#   }
# }
#
# 
# plot.dbscan.eps(PCs.sub, 5)
pdf("dbscan_distplot.pdf")
par(mfrow=c(1,3), mai=c(.75,.75,.4,.3))
for(i in seq_along(PCs.sub)){
  dbscan::kNNdistplot(PCs.sub[[i]], 4)
  title( main=paste(pc.ind[i], "Components Kept"), line=.5)
  title(main = "Distance to Nearest Neighbor Across PC Spaces", outer=T, line=-1)
}
dev.off()

scan.grid.5 = expand.grid(eps = seq(380, 600, 5), 
                          minPts = seq(5, 50, by = 5)
                          )
scan.grid.20 = expand.grid(eps = seq(800, 1500, 5),
                           minPts= seq(5, 50, 5)
                           )
scan.grid.40 = expand.grid(eps = seq(1000,1800, 5),
                           minPts= seq(5, 50, 5)
                           )
scan.grid = list(scan.grid.5, scan.grid.20, scan.grid.40)





#function for computing CH index
ch.index3 = function(data, k, twss, bss){
  
  n = nrow(data)
  ch=(bss/(k-1))/(twss/(n-k))

  return(ch)
}

ch.all = list()
for(g in seq_along(scan.grid)){
  
  grid = scan.grid[[g]]
  data = PCs.sub[[g]]
  ch = vector()
  
    for(parm in seq_len(nrow(grid))){
      
    scan = dbscan(data,
                  eps = grid[parm, "eps"], 
                  minPts = grid[parm, "minPts"])
    
    cluster = scan$cluster[scan$cluster != 0]
    ncluster = length(unique(cluster))
    
    if(ncluster <=1){ ch[parm] = NA; next} # CH undefined for k=1
    
    df.scan = data[scan$cluster != 0, ]
    ss = clust.SS(data=df.scan, cluster = cluster)
    ch[parm] = ch.index3(data=df.scan, k=ncluster, twss=ss$twss, bss=ss$bss)
    }
  
  ch.all[[g]] = ch
}

sapply(ch.all, which.max)
scan.opt = list()

for(i in seq_along(ch.all)){
  scan.opt[[i]] = scan.grid[[i]][which.max(ch.all[[i]]),]
}

names(scan.opt) = names(PCs.sub)
print(scan.opt)

  scan.5 = dbscan(PCs.sub[["y.pc.5"]], 
                eps=scan.opt[["y.pc.5"]][,"eps"],
                minPts = scan.opt[["y.pc.5"]][,"minPts"]
                )

scan.20 = dbscan(PCs.sub[["y.pc.20"]], 
                eps=scan.opt[["y.pc.20"]][,"eps"],
                minPts = scan.opt[["y.pc.20"]][,"minPts"]
                )

scan.40 = dbscan(PCs.sub[["y.pc.40"]], 
                eps=scan.opt[["y.pc.40"]][,"eps"],
                minPts = scan.opt[["y.pc.40"]][,"minPts"]
                )

scan.40$cluster

pdf("dbscan_pcspace.pdf")
plot(PCs[,1], PCs[,2], col=scan.40$cluster + 1)
dev.off()
sink()