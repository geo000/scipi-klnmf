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
void em_klnmf(const arma::mat& V, arma::mat& W, arma::mat& H,
              arma::rowvec& wcolsum, arma::vec& hrowsum, int subprob_iter);
void scipi_klnmf(const arma::mat& V, arma::mat& W, arma::mat& H,
                 const arma::rowvec& vcolsum, const arma::vec& vrowsum,
                 arma::rowvec& wcolsum, arma::vec& hrowsum, int subprob_iter);
void stochastic_scipi_dense_klnmf(const arma::mat& V,
                                  arma::mat& W, arma::mat& H,
                                  const arma::rowvec& vcolsum, const arma::vec& vrowsum,
                                  arma::rowvec& wcolsum, arma::vec& hrowsum, int subprob_iter,
                                  int batch_size, int epoch_length);
void ccd_klnmf(const arma::mat& V, arma::mat& W, arma::mat& H, arma::mat& B, int subprob_iter);
// ---------------------
void row_full_update_dense(const arma::mat& V, const arma::mat& W, arma::mat& H);
void col_full_update_dense(const arma::mat& V, arma::mat& W, const arma::mat& H);
void colmat_single_epoch_dense(const arma::mat& V,
                               arma::mat& W, const arma::mat& H,
                               int batch_size, int epoch_length);
// ---------------------
void rowmat_stochastic_grad_dense(const arma::mat& V, const arma::mat& W, const arma::mat& H,
                                  const arma::mat& Hold, arma::mat& Hgrad, int batch_size);
void colmat_stochastic_grad_dense(const arma::mat& V, const arma::mat& W, const arma::mat& H,
                                  const arma::mat& Wold, arma::mat& Wgrad, int batch_size);
// ---------------------
void compute_obj_dense(const arma::mat& V, const arma::mat& A, double& obj);
// ---------------------
double shift_parameter    = 0.0;
double step_size          = 1.0;
int n;
int m;
int d;
double epstol             = 1e-15;
// ---------------------

// FUNCTION DEFINITIONS
// --------------------
// --------------------
// [[Rcpp::export]]
Rcpp::List dense_klnmf(const arma::mat& V, arma::mat& W, arma::mat& H,
                       const arma::rowvec& vcolsum, const arma::vec& vrowsum,
                       std::string method,
                       int max_iter, int init_em_iter, int subprob_iter,
                       int batch_size, int epoch_length,
                       double shift_parameter_input, double step_size_input,
                       int reportnum, bool verbose) {
  
  n                       = V.n_rows;
  m                       = V.n_cols;
  d                       = W.n_cols;
  int report_i            = 0;
  int i;
  
  // precalculate
  double vsum             = arma::sum(vrowsum);
  arma::mat A             = V;
  arma::rowvec wcolsum(d);
  arma::vec hrowsum(d);
  shift_parameter         = shift_parameter_input;
  step_size               = step_size_input;
  
  // set timer and objective
  auto start_time = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> delta = (std::chrono::steady_clock::now() - start_time);
  arma::vec timing(init_em_iter + max_iter + 1, arma::fill::zeros);
  arma::vec tim(init_em_iter + max_iter + 1, arma::fill::zeros);
  arma::vec obj(init_em_iter + max_iter + 1, arma::fill::zeros);
  
  for (i = 0; i < init_em_iter; i++) {
    start_time            = std::chrono::steady_clock::now();
    em_klnmf(V, W, H, wcolsum, hrowsum, subprob_iter);
    delta                 = (std::chrono::steady_clock::now() - start_time);
    timing[i]            += static_cast<double>(delta.count());
    
    if (std::remainder(i, reportnum) == 0) {
      A                   = V / (W * H);
      compute_obj_dense(V, A, obj(report_i));
      if (verbose) {
        Rprintf("iteration %3d: %3e\n", i , obj(report_i));
      }
      report_i++;
    }
    
    // -------------------------------------------------------
    // check user interrupt
    // -------------------------------------------------------
    Rcpp::checkUserInterrupt();
  }
  
  arma::mat B;
  if (method == std::string("ccd")) {
    B                     = W * H;
  }
  
  for (; i < max_iter + init_em_iter; i++) {
    start_time            = std::chrono::steady_clock::now();
    if (method == std::string("scipi")) {
      scipi_klnmf(V, W, H, vcolsum, vrowsum, wcolsum, hrowsum, subprob_iter);
    } else if (method == std::string("sscipi")) {
      stochastic_scipi_dense_klnmf(V, W, H, vcolsum, vrowsum, wcolsum, hrowsum, subprob_iter, batch_size, epoch_length);
    } else if (method == std::string("em")) {
      em_klnmf(V, W, H, wcolsum, hrowsum, subprob_iter);
    } else if (method == std::string("ccd")) {
      ccd_klnmf(V, W, H, B, subprob_iter);
      B                   = W * H;
    }
    delta                 = (std::chrono::steady_clock::now() - start_time);
    timing(i)            += static_cast<double>(delta.count());
    tim(report_i)        += timing(i);
    
    if (std::remainder(i, reportnum) == 0) {
      if (method == std::string("ccd")) {
        A                   = V / B;
        obj(i)             += arma::accu(B) - vsum;
      } else {
        A                   = V / (W * H);
      }
      compute_obj_dense(V, A, obj(report_i));
      
      if (verbose) {
        Rprintf("iteration %3d: %3e\n", i , obj(report_i));
      }
      
      report_i++;
    }
    
    // -------------------------------------------------------
    // check user interrupt
    // -------------------------------------------------------
    Rcpp::checkUserInterrupt();
  }
  
  // report last results
  A                         = V / (W * H);
  compute_obj_dense(V, A, obj(report_i));
  
  if (verbose) {
    Rprintf("terminated at iteration %3d: %3e\n", i-1 , obj(report_i));
  }

  
  if (report_i > 0) {
    return Rcpp::List::create(Rcpp::Named("obj")    = obj.subvec(0,report_i-1),
                              Rcpp::Named("t")      = tim.subvec(0,report_i-1),
                              Rcpp::Named("fobj")   = obj(report_i),
                              Rcpp::Named("timing") = timing);
  }
  
  return Rcpp::List::create  (Rcpp::Named("obj")    = std::string("not requested"),
                              Rcpp::Named("t")      = std::string("not requested"),
                              Rcpp::Named("fobj")   = obj(report_i),
                              Rcpp::Named("timing") = timing);
}

// [[Rcpp::export]]
Rcpp::List scipi_dense_klnmf(const arma::mat& V, arma::mat& W, arma::mat& H,
                             const arma::rowvec& vcolsum, const arma::vec& vrowsum,
                             int max_iter, int init_em_iter, int subprob_iter,
                             double shift_parameter_input, double step_size_input,
                             bool verbose) {
  
  n                       = V.n_rows;
  m                       = V.n_cols;
  d                       = W.n_cols;
  int i;
  
  // set timer
  auto start_time = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> delta = (std::chrono::steady_clock::now() - start_time);
  arma::vec timing(init_em_iter + max_iter, arma::fill::zeros);
  
  arma::mat A             = V;
  arma::rowvec wcolsum(d);
  arma::vec hrowsum(d);
  shift_parameter         = shift_parameter_input;
  step_size               = step_size_input;
  
  arma::vec obj(max_iter + init_em_iter, arma::fill::zeros);
  
  for (i = 0; i < init_em_iter; i++) {
    start_time            = std::chrono::steady_clock::now();
    em_klnmf(V, W, H, wcolsum, hrowsum, subprob_iter);
    delta                 = (std::chrono::steady_clock::now() - start_time);
    timing[i]            += static_cast<double>(delta.count());
    A                     = V / (W * H);
    compute_obj_dense(V, A, obj(i));
    if (verbose & (std::remainder(i, 100) == 0)) {
      Rprintf("iteration %3d: %3e\n",i,obj(i));
    }
    
    // -------------------------------------------------------
    // check user interrupt
    // -------------------------------------------------------
    Rcpp::checkUserInterrupt();
  }
  
  for (; i < max_iter + init_em_iter; i++) {
    start_time            = std::chrono::steady_clock::now();
    scipi_klnmf(V, W, H, vcolsum, vrowsum, wcolsum, hrowsum, subprob_iter);
    delta                 = (std::chrono::steady_clock::now() - start_time);
    timing[i]            += static_cast<double>(delta.count());
    A                     = V / (W * H);
    compute_obj_dense(V, A, obj(i));
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

// [[Rcpp::export]]
Rcpp::List stochastic_scipi_dense_klnmf(const arma::mat& V, arma::mat& W, arma::mat& H,
                                        const arma::rowvec& vcolsum, const arma::vec& vrowsum,
                                        int max_iter, int init_em_iter, int subprob_iter,
                                        int batch_size, int epoch_length,
                                        double shift_parameter_input, double step_size_input, bool verbose) {
  
  n                       = V.n_rows;
  m                       = V.n_cols;
  d                       = W.n_cols;
  int i;
  
  // set timer
  auto start_time = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> delta = (std::chrono::steady_clock::now() - start_time);
  arma::vec timing(init_em_iter + max_iter, arma::fill::zeros);
  
  arma::mat A             = V;
  arma::rowvec wcolsum(d);
  arma::vec hrowsum(d);
  shift_parameter         = shift_parameter_input;
  step_size               = step_size_input;
  
  arma::vec obj(max_iter + init_em_iter, arma::fill::zeros);
  
  for (i = 0; i < init_em_iter; i++) {
    start_time            = std::chrono::steady_clock::now();
    em_klnmf(V, W, H, wcolsum, hrowsum, subprob_iter);
    delta                 = (std::chrono::steady_clock::now() - start_time);
    timing[i]            += static_cast<double>(delta.count());
    A                     = V / (W * H);
    compute_obj_dense(V, A, obj(i));
    if (verbose & (std::remainder(i, 100) == 0)) {
      Rprintf("iteration %3d: %3e\n",i,obj(i));
    }
    
    // -------------------------------------------------------
    // check user interrupt
    // -------------------------------------------------------
    Rcpp::checkUserInterrupt();
  }
  
  for (; i < max_iter + init_em_iter; i++) {
    start_time            = std::chrono::steady_clock::now();
    stochastic_scipi_dense_klnmf(V, W, H, vcolsum, vrowsum, wcolsum, hrowsum, subprob_iter, batch_size, epoch_length);
    delta                 = (std::chrono::steady_clock::now() - start_time);
    timing[i]            += static_cast<double>(delta.count());
    A                     = V / (W * H);
    compute_obj_dense(V, A, obj(i));
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

void stochastic_scipi_dense_klnmf(const arma::mat& V,
                                  arma::mat& W, arma::mat& H,
                                  const arma::rowvec& vcolsum, const arma::vec& vrowsum,
                                  arma::rowvec& wcolsum, arma::vec& hrowsum, int subprob_iter,
                                  int batch_size, int epoch_length) {
  
  // run subprob for H
  for (int j = 0; j < subprob_iter; j++) {
    wcolsum                 = arma::sum(W, 0);
    W.each_row()           /= wcolsum;
    H.each_col()           %= wcolsum.t();
    H.each_row()           /= vcolsum;
    row_full_update_dense(V, W, H);
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
    colmat_single_epoch_dense(V, W, H, batch_size, epoch_length);
    W.each_row()           /= hrowsum.t();
    W.each_col()           %= vrowsum;
    H.each_col()           %= hrowsum;
  }
}

void em_klnmf(const arma::mat& V, arma::mat& W, arma::mat& H,
              arma::rowvec& wcolsum, arma::vec& hrowsum, int subprob_iter) {
  
  // run subprob for H
  for (int j = 0; j < subprob_iter; j++) {
    wcolsum                 = arma::sum(W, 0);
    H                      %= W.t() * (V / (W * H));
    H.each_col()           /= wcolsum.t();
  }
  
  // run subprob for W
  for (int j = 0; j < subprob_iter; j++) {
    hrowsum                 = arma::sum(H, 1);
    W                      %= (V / (W * H)) * H.t();
    W.each_row()           /= hrowsum.t();
  }
}

void scipi_klnmf(const arma::mat& V, arma::mat& W, arma::mat& H,
                 const arma::rowvec& vcolsum, const arma::vec& vrowsum,
                 arma::rowvec& wcolsum, arma::vec& hrowsum, int subprob_iter) {
  
  // run subprob for H
  for (int j = 0; j < subprob_iter; j++) {
    wcolsum                 = arma::sum(W, 0);
    W.each_row()           /= wcolsum;
    H.each_col()           %= wcolsum.t();
    H.each_row()           /= vcolsum;
    row_full_update_dense(V, W, H);
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
    col_full_update_dense(V, W, H);
    W.each_row()           /= hrowsum.t();
    W.each_col()           %= vrowsum;
    H.each_col()           %= hrowsum;
  }
}

void ccd_klnmf(const arma::mat& V, arma::mat& W, arma::mat& H, arma::mat& B, int subprob_iter) {
  
  double a;
  double b;
  double c;
  double diff;
  double e;
  
  // run subprob for H
  for (int u = 0; u < subprob_iter; u++) {
    for (int j = 0; j < m; j++) {
      for (int k = 0; k < d; k++) {
        a                    = arma::dot(1.0 - V.col(j) / (B.col(j) + epstol), W.col(k));
        b                    = arma::dot(V.col(j), arma::square(W.col(k) / (B.col(j) + epstol)));
        e                    = H(k,j);
        c                    = std::max(e - a/b + epstol, epstol);
        diff                 = c - e;
        H(k,j)               = c;
        B.col(j)            += diff * W.col(k);
      }
    }
  }
  
  // run subprob for W
  for (int u = 0; u < subprob_iter; u++) {
    for (int i = 0; i < n; i++) {
      for (int k = 0; k < d; k++) {
        a                    = arma::dot(1.0 - V.row(i) / (B.row(i) + epstol), H.row(k));
        b                    = arma::dot(V.row(i), arma::square(H.row(k) / (B.row(i) + epstol)));
        e                    = W(i,k);
        c                    = std::max(e - a/b + epstol, epstol);
        diff                 = c - e;
        W(i,k)               = c;
        B.row(i)            += diff * H.row(k);
      }
    }
  }
}

void row_full_update_dense(const arma::mat& V, const arma::mat& W, arma::mat& H) {
  H                       %= arma::square((1.0 + shift_parameter - step_size) + step_size * (W.t() * (V / (W * H))));
  H                        = arma::normalise(H, 1, 0);
}

void col_full_update_dense(const arma::mat& V, arma::mat& W, const arma::mat& H) {
  W                       %= arma::square((1.0 + shift_parameter - step_size) + step_size * ((V / (W * H)) * H.t()));
  W                        = arma::normalise(W, 1, 1);
}

void compute_obj_dense(const arma::mat& V, const arma::mat& A, double& obj) {
  obj                     += arma::dot(V, arma::log(A + epstol));
}

void rowmat_single_epoch_dense(const arma::mat& V,
                               const arma::mat& W, arma::mat& H,
                               int batch_size, int epoch_length) {
  
  double tempdouble         = static_cast<double>(batch_size) / static_cast<double>(n);
  arma::mat Hold            = H;
  arma::mat Hgrad           = W.t() * (V / (W * Hold)) * tempdouble;
  arma::mat Hgrad2;
  
  for (int i = 0; i < epoch_length; i++) {
    Hgrad2                  = Hgrad;
    rowmat_stochastic_grad_dense(V, W, H, Hold, Hgrad2, batch_size);
    H                      %= arma::square(Hgrad2);
    H                       = arma::normalise(H, 1, 0);
  }
}

void rowmat_stochastic_grad_dense(const arma::mat& V, const arma::mat& W, const arma::mat& H,
                                  const arma::mat& Hold, arma::mat& Hgrad, int batch_size) {
  // -------------------------------------------------------
  // STEP 1: sample rows of V
  // -------------------------------------------------------
  arma::uvec sampled_index(n,arma::fill::zeros);
  for (int j = 0; j < batch_size; j++) {
    sampled_index(std::rand() % n) = 1;
  }
  arma::uvec bin_index = find(sampled_index);
  
  // -------------------------------------------------------
  // STEP 2: compute svrg gradient
  // -------------------------------------------------------
  Hgrad               += W.rows(bin_index).t() * (V.rows(bin_index) / (W.rows(bin_index) * H)) -
    W.rows(bin_index).t() * (V.rows(bin_index) / (W.rows(bin_index) * Hold));
}

void colmat_single_epoch_dense(const arma::mat& V,
                               arma::mat& W, const arma::mat& H,
                               int batch_size, int epoch_length) {
  
  double tempdouble         = static_cast<double>(batch_size) / static_cast<double>(m);
  arma::mat Wold            = W;
  arma::mat Wgrad           = (V / (Wold * H)) * (H.t() * tempdouble);
  arma::mat Wgrad2;
  
  for (int i = 0; i < epoch_length; i++) {
    Wgrad2                  = Wgrad;
    colmat_stochastic_grad_dense(V, W, H, Wold, Wgrad2, batch_size);
    W                      %= arma::square(Wgrad2);
    W                       = arma::normalise(W, 1, 1);
  }
}

void colmat_stochastic_grad_dense(const arma::mat& V, const arma::mat& W, const arma::mat& H,
                                  const arma::mat& Wold, arma::mat& Wgrad, int batch_size) {
  // -------------------------------------------------------
  // STEP 1: sample rows of V
  // -------------------------------------------------------
  arma::uvec sampled_index(m,arma::fill::zeros);
  for (int j = 0; j < batch_size; j++) {
    sampled_index(std::rand() % m) = 1;
  }
  arma::uvec bin_index = find(sampled_index);
  
  // -------------------------------------------------------
  // STEP 2: compute svrg gradient
  // -------------------------------------------------------
  Wgrad               += (V.cols(bin_index) / (W * H.cols(bin_index))) * H.cols(bin_index).t() -
    (V.cols(bin_index) / (Wold * H.cols(bin_index))) * H.cols(bin_index).t();
}