// This is included to suppress the warnings from solve() when the
// system is singular or close to singular.
#define ARMA_DONT_PRINT_ERRORS

#include <RcppArmadillo.h>
#include <Rcpp/Benchmark/Timer.h>

// This depends statement is needed to tell R where to find the
// additional header files.
// [[Rcpp::plugins(cpp11)]] 
//
// [[Rcpp::depends(RcppArmadillo)]]
//

// FUNCTION DECLARATIONS
// ---------------------
double shift_parameter    = 0.0;
double step_size          = 1.0;
int n;
int m;
int d;
int N;
double alpha;
double epstol              = 1e-15;
// ---------------------
void em_sparse_klnmf_vec(const arma::uvec& rowind, const arma::uvec& colind, const arma::vec& val,
                         arma::mat& W, arma::mat& H,
                         arma::rowvec& wcolsum, arma::vec& hrowsum, int subprob_iter);
void scipi_sparse_klnmf_vec(const arma::uvec& rowind, const arma::uvec& colind, const arma::vec& val,
                            arma::mat& W, arma::mat& H,
                            const arma::rowvec& vcolsum, const arma::vec& vrowsum,
                            arma::rowvec& wcolsum, arma::vec& hrowsum, int subprob_iter);
// ---------------------
int rowvec_single_epoch(const arma::uvec& rowind, const arma::uvec& colind, const arma::vec& val,
                        const arma::mat& W, arma::mat& H,
                        int batch_size, int epoch_length);
int colvec_single_epoch(const arma::uvec& rowind, const arma::uvec& colind, const arma::vec& val,
                        arma::mat& W, const arma::mat& H,
                        int batch_size, int epoch_length);
// ---------------------
void rowvec_full_grad(const arma::uvec& rowind, const arma::uvec& colind, const arma::vec& val,
                      const arma::mat& W, const arma::mat& H, arma::mat& Hgrad);
void colvec_full_grad(const arma::uvec& rowind, const arma::uvec& colind, const arma::vec& val,
                      const arma::mat& W, const arma::mat& H, arma::mat& Wgrad);
// ---------------------
void compute_obj_vec(const arma::uvec& rowind, const arma::uvec& colind, const arma::vec& val,
                     const arma::mat& W, const arma::mat& H, double& obj);

// FUNCTION DEFINITIONS
// --------------------
// [[Rcpp::export]]
Rcpp::List sparse_scipi_klnmf(const arma::uvec& rowind, const arma::uvec& colind, const arma::vec& val,
                              arma::mat& W, arma::mat& H,
                              const arma::rowvec& vcolsum, const arma::vec& vrowsum,
                              int max_iter, int init_em_iter, int subprob_iter,
                              double alpha_input,
                              double shift_parameter_input, double step_size_input,
                              bool verbose){
  
  // -------------------------------------------------------
  // STEP 0: set miscellaneous things
  // -------------------------------------------------------
  n                       = W.n_rows;
  m                       = H.n_cols;
  d                       = W.n_cols;
  N                       = rowind.n_elem;
  int i;
  
  // pre-allocate
  arma::rowvec wcolsum(d);
  arma::vec hrowsum(d);
  arma::vec obj(max_iter + init_em_iter, arma::fill::zeros);
  shift_parameter         = shift_parameter_input;
  step_size               = step_size_input;
  alpha                   = alpha_input;
  
  // set timer
  auto start_time = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> delta = (std::chrono::steady_clock::now() - start_time);
  arma::vec timing(init_em_iter + max_iter, arma::fill::zeros);
  
  // -------------------------------------------------------
  // STEP 2: iterate em for initialization
  // -------------------------------------------------------
  for (i = 0; i < init_em_iter; i++) {
    
    // -------------------------------------------------------
    // STEP 2.1: em update
    // -------------------------------------------------------
    start_time            = std::chrono::steady_clock::now();
    em_sparse_klnmf_vec(rowind, colind, val, W, H, wcolsum, hrowsum, subprob_iter);
    delta                 = (std::chrono::steady_clock::now() - start_time);
    timing[i]            += static_cast<double>(delta.count());
    // -------------------------------------------------------
    // STEP 2.2: compute objective value
    // -------------------------------------------------------
    compute_obj_vec(rowind, colind, val, W, H, obj(i));
    // -------------------------------------------------------
    // STEP 2.3: report
    // -------------------------------------------------------
    if (verbose & (std::remainder(i, 100) == 0)) {
      Rprintf("iteration %3d: %3e\n",i,obj(i));
    }
    // -------------------------------------------------------
    // check user interrupt
    // -------------------------------------------------------
    Rcpp::checkUserInterrupt();
  }
  
  // -------------------------------------------------------
  // STEP 3: iterate single-epoch updates
  // -------------------------------------------------------
  for (; i < max_iter + init_em_iter; i++) {
    // -------------------------------------------------------
    // STEP 3.1: stochastic update
    // -------------------------------------------------------
    start_time            = std::chrono::steady_clock::now();
    scipi_sparse_klnmf_vec(rowind, colind, val, W, H, vcolsum, vrowsum, wcolsum, hrowsum, subprob_iter);
    delta                 = (std::chrono::steady_clock::now() - start_time);
    timing[i]            += static_cast<double>(delta.count());
    // -------------------------------------------------------
    // STEP 3.2: compute objective value
    // -------------------------------------------------------
    compute_obj_vec(rowind, colind, val, W, H, obj(i));
    // -------------------------------------------------------
    // STEP 3.3: report
    // -------------------------------------------------------
    if (verbose & (std::remainder(i, 100) == 0)) {
      Rprintf("iteration %3d: %3e\n",i,obj(i));
    }
    
    // -------------------------------------------------------
    // check user interrupt
    // -------------------------------------------------------
    Rcpp::checkUserInterrupt();
  }
  
  if (verbose) {
    Rprintf("iteration %3d: %3e\n",i,obj(i-1));
  }
  
  return Rcpp::List::create(Rcpp::Named("obj")    = obj,
                            Rcpp::Named("timing") = timing);
}


// --------------------
void scipi_sparse_klnmf_vec(const arma::uvec& rowind, const arma::uvec& colind, const arma::vec& val,
                            arma::mat& W, arma::mat& H,
                            const arma::rowvec& vcolsum, const arma::vec& vrowsum,
                            arma::rowvec& wcolsum, arma::vec& hrowsum, int subprob_iter) {
  
  // run subprob for H
  arma::mat Hgrad(d,m);
  arma::mat Wgrad(n,d);
  
  // run subprob for H
  for (int j = 0; j < subprob_iter; j++) {
    wcolsum                 = arma::sum(W, 0);
    W.each_row()           /= wcolsum;
    H.each_col()           %= wcolsum.t();
    H.each_row()           /= vcolsum;
    Hgrad.fill(0);
    rowvec_full_grad(rowind, colind, val, W, H, Hgrad);
    H                      %= arma::square((1.0 + shift_parameter - step_size) + step_size * Hgrad);
    H                       = arma::normalise(H, 1, 0);
    H.each_col()           /= wcolsum.t();
    H.each_row()           %= vcolsum;
    W.each_row()           %= wcolsum;
  }
  
  // run subprob for W
  for (int j = 0; j < subprob_iter; j++) {
    hrowsum                 = arma::sum(H, 1);
    H.each_col()           /= hrowsum;
    W.each_row()           %= hrowsum.t();
    W.each_col()           /= vrowsum;
    Wgrad.fill(0);
    colvec_full_grad(rowind, colind, val, W, H, Wgrad);
    W                      %= arma::square((1.0 + shift_parameter - step_size) + step_size * Wgrad);
    W                       = arma::normalise(W, 1, 1);
    W.each_row()           /= hrowsum.t();
    W.each_col()           %= vrowsum;
    H.each_col()           %= hrowsum;
  }
}

// --------------------
void rowvec_full_grad(const arma::uvec& rowind, const arma::uvec& colind, const arma::vec& val,
                      const arma::mat& W, const arma::mat& H, arma::mat& Hgrad) {
  
  for (int j = 0; j < N; j++) {
    Hgrad.col(colind(j)) += (val(j) / arma::dot(W.row(rowind(j)).t(), H.col(colind(j)))) * W.row(rowind(j)).t();
  }
}

// --------------------
void colvec_full_grad(const arma::uvec& rowind, const arma::uvec& colind, const arma::vec& val,
                      const arma::mat& W, const arma::mat& H, arma::mat& Wgrad) {
  
  for (int j = 0; j < N; j++) {
    Wgrad.row(rowind(j)) += (val(j) / arma::dot(W.row(rowind(j)).t(), H.col(colind(j)))) * H.col(colind(j)).t();
  }
}

// --------------------
void compute_obj_vec(const arma::uvec& rowind, const arma::uvec& colind, const arma::vec& val,
                     const arma::mat& W, const arma::mat& H, double& obj) {
  
  for (int j = 0; j < N; j++) {
    obj                  += val(j) * log(val(j) / arma::dot(W.row(rowind(j)).t(), H.col(colind(j))) + epstol);
  }
}