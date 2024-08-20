library(bnlearn)

seed_list = c(357,470,2743,4951,5088,5657,5852,6049,6659,9076)
names = c('sangiovese','mehra','healthcare','ecoli70','magic-niab','magic-irri','arth150')
for (seed in seed_list){
  set.seed(seed)
  for (name in names){
    load(paste0("/Downloads/",name,".rda"))
    Xtest <- bnlearn::rbn(bn, n=50000)
    saveRDS(Xtest,paste0(name,'_',seed,'.rds'))
    dag <- bnlearn::amat(bn)
    saveRDS(dag, paste0(name,'_','DAG.rds'))
  }
}