

if (!require("BiocManager", quietly = TRUE)){
    install.packages("BiocManager")
    library("BiocManager")
}
# BiocManager::install(version = "3.17")
bioc_packages = c('graph', 'RBGL')

## Now load or install&load all
bioc_package.check <- lapply(
  bioc_packages,
  FUN = function(x) {
    if (!require(x, character.only = TRUE)) {
      BiocManager::install(x)
      library(x, character.only = TRUE)
    }
  }
)

## First specify the packages of interest
packages = c('pcalg',
            'kpcalg',
            # 'bnlearn',
            # 'sparsebn',
            # 'SID',
            # 'D2C',
            'devtools')

## Now load or install&load all
package.check <- lapply(
  packages,
  FUN = function(x) {
    if (!require(x, character.only = TRUE)) {
      install.packages(x, dependencies = TRUE, repos = "http://cloud.r-project.org")
      library(x, character.only = TRUE)
    }
  }
)

## Check and install packages from github
if (!require('RCIT', character.only = TRUE)) {
      install_github("Diviyan-Kalainathan/RCIT")
      library('RCIT', character.only = TRUE)
    }
if (!require('CAM', character.only = TRUE)) {
      install_github("cran/CAM")
      library('CAM', character.only = TRUE)
    }
if (!require('SID', character.only = TRUE)) {
      install_github("cran/SID")
      library('SID', character.only = TRUE)
    }

    
    

