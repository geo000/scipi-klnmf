#' @title SCI-PI for KL-NMF (SCale Invariant Power Iteration for KL-divergence NMF)
#' 
#' @description
#' 
#' @details Add details here.
#'
#' @param X Describe X here. Sparse matrix? Dense?
#' 
#' @param y Describe y here.
#' 
#' @return Describe return value here.
#' 
#' @useDynLib scipi.klnmf
#' 
#' @importFrom Rcpp evalCpp
#' 
#' @examples
#' 
#' @export
#' 
scipi.klnmf <- function(V, k = 1, init = NULL, 
                        method = c("scipi", "em"), 
                        max.iter = 100, init.em.iter = 5, subprob.iter = 1,
                        report.num = 1,
                        seed = 1,
                        verbose = FALSE) {
  
  # set.seed
  set.seed(seed)
  
  # step 0
  is.V.sparse <- is.matrix(V)
  n           <- dim(V)[1]
  m           <- dim(V)[2]
  method      <- match.arg(method)
  
  if (any(V < 0)) {
    stop("The input matrix V should have non-negative entries")
  }
  
  vcolsum     <- colSums(V)
  vrowsum     <- rowSums(V)
  
  # set initialization
  if (is.null(init)) {
    init.W    <- matrix(runif(n*k), n, k)
    init.H    <- matrix(runif(k*m), k, m)
  }
  
  # initialize W and H with init.em.iter steps
  out         <- dense_klnmf(V, init.W, init.H, vcolsum, vrowsum,
                             method,
                             0, init.em.iter, subprob.iter, 
                             0, 0, 0.0, 1.0, report.num, FALSE)
  
  # save results for initialization
  res         <- list(init.W  = init.W,
                      init.H  = init.H,
                      iobj    = out$fobj)

  # force making copies
  res$init.W[1,1]  <- res$init.W[1,1] + 0.0
  res$init.H[1,1]  <- res$init.H[1,1] + 0.0
  
  # main function: run method for max.iter steps
  out         <- dense_klnmf(V, init.W, init.H, vcolsum, vrowsum,
                             method,
                             max.iter, 0, subprob.iter, 
                             0, 0, 0.0, 1.0, report.num, verbose)
  
  # summarize and return results
  res$fobj    <- drop(out$fobj)
  res$obj     <- drop(out$obj)
  res$objall  <- c(res$iobj, res$obj, res$fobj)
  res$t       <- drop(out$t)
  res$timing  <- drop(out$timing)
  res$W       <- init.W
  res$H       <- init.H
  
  # remove possibly large temporary output
  rm(out)
  
  return (res)
}