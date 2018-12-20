// This file was generated by Rcpp::compileAttributes
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// migrad_cpp
Rcpp::List migrad_cpp(Rcpp::NumericVector par, Rcpp::Function f_R, Rcpp::NumericVector lower, Rcpp::NumericVector higher, Rcpp::List control);
RcppExport SEXP migrad_migrad_cpp(SEXP parSEXP, SEXP f_RSEXP, SEXP lowerSEXP, SEXP higherSEXP, SEXP controlSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type par(parSEXP);
    Rcpp::traits::input_parameter< Rcpp::Function >::type f_R(f_RSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type lower(lowerSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type higher(higherSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type control(controlSEXP);
    __result = Rcpp::wrap(migrad_cpp(par, f_R, lower, higher, control));
    return __result;
END_RCPP
}
// migrad_grad_cpp
Rcpp::List migrad_grad_cpp(Rcpp::NumericVector par, Rcpp::Function f_R, Rcpp::Function g_R, Rcpp::NumericVector lower, Rcpp::NumericVector higher, Rcpp::List control);
RcppExport SEXP migrad_migrad_grad_cpp(SEXP parSEXP, SEXP f_RSEXP, SEXP g_RSEXP, SEXP lowerSEXP, SEXP higherSEXP, SEXP controlSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type par(parSEXP);
    Rcpp::traits::input_parameter< Rcpp::Function >::type f_R(f_RSEXP);
    Rcpp::traits::input_parameter< Rcpp::Function >::type g_R(g_RSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type lower(lowerSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type higher(higherSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type control(controlSEXP);
    __result = Rcpp::wrap(migrad_grad_cpp(par, f_R, g_R, lower, higher, control));
    return __result;
END_RCPP
}
