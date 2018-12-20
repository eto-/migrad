/*
 * to update the RcppExports use
 * compileAttributes(pkgdir = ".", verbose = getOption("verbose"))
 */
#include <Minuit2/FCNGradientBase.h>
#include <Minuit2/MnMigrad.h>
#include <Minuit2/MnUserParameters.h>
#include <Minuit2/MnHesse.h>
#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/MnPrint.h>
#include <Rcpp.h>
#include <string>


class fcn {
  public:
    fcn (Rcpp::Function& f, Rcpp::Function& g, bool check_gradient): f_(f), g_(g), check_gradient_(check_gradient) {} 

    double value (const std::vector<double>& p) const { return Rcpp::as<double>(f_(p)); }

    std::vector<double> gradient (const std::vector<double>& p) const { return Rcpp::as<std::vector<double>>(g_(p)); }

    bool check_gradient () const { return check_gradient_; }

    double up() const { return 0.5; }
  private:
    Rcpp::Function& f_, g_;
    bool check_gradient_;
};

class fcn_base: public ROOT::Minuit2::FCNBase { 
  public:
    fcn_base (Rcpp::Function& f, bool check_gradient): f_(f, f, check_gradient) {} 
    virtual ~fcn_base () {}

    virtual double Up() const { return f_.up (); }
    virtual double operator() (const std::vector<double>& p) const { return f_.value (p); }
  private:
    fcn f_;
};

class fcn_grad: public ROOT::Minuit2::FCNGradientBase { 
  public:
    fcn_grad (Rcpp::Function& f, Rcpp::Function& g, bool check_gradient): f_(f, g, check_gradient) {}
    virtual ~fcn_grad () {}

    virtual double Up() const { return f_.up (); }
    virtual double operator() (const std::vector<double>& p) const { return f_.value (p); }
    virtual std::vector<double> Gradient (const std::vector<double>& p) const { return f_.gradient (p); }
    virtual bool CheckGradient () const { return f_.check_gradient (); }

  private:
    fcn f_;
};

std::vector<std::string> v = { "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t" };
ROOT::Minuit2::MnUserParameters pack_parameters (Rcpp::NumericVector& par, Rcpp::NumericVector& lower, Rcpp::NumericVector& upper) {
  ROOT::Minuit2::MnUserParameters p;
  for (int i=0; i < par.size (); i++) {
    p.Add (v[i], par[i], par[i]/1000);
    if (Rcpp::NumericVector::is_na (lower[i]) || Rcpp::NumericVector::is_na (upper[i])) continue;
    if (!R_finite (lower[i]) || !R_finite (upper[i])) continue;
    p.SetLimits (i, lower[i], upper[i]);
  }

  return p;
}

Rcpp::List pack_results (ROOT::Minuit2::FunctionMinimum& m) {
  double value = m.Fval ();
  int convergence = m.IsValid () ? 0 : 1;
  int counts = m.NFcn ();
  int n = m.UserCovariance ().Nrow ();
  Rcpp::NumericMatrix cov(n);
  for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) cov(i,j) = m.UserCovariance ()(i,j); 

  return Rcpp::List::create(Rcpp::Named("par") = m.UserParameters ().Params (), 
			    Rcpp::Named("value") = value, 
			    Rcpp::Named("counts") = counts, 
			    Rcpp::Named("convergence") = convergence, 
			    Rcpp::Named("covariace") = cov);
}

//template <typename T> T def_v (Rcpp::List& l, const std::string& n, T def) {
//  if (l.containsElementNamed(n.c_str ()))
//    return Rcpp::as<T>(l[n.c_str ()]);
//  return def;
//}

// [[Rcpp::export]]
Rcpp::List migrad_cpp (Rcpp::NumericVector par, Rcpp::Function f_R, Rcpp::NumericVector lower, Rcpp::NumericVector upper, Rcpp::List control) {
  ROOT::Minuit2::MnUserParameters p = pack_parameters(par, lower, upper);

  fcn_base f(f_R, Rcpp::as<bool>(control["check.gradient"]));

  ROOT::Minuit2::MnMigrad migrad(f, p, ROOT::Minuit2::MnStrategy(Rcpp::as<int>(control["strategy"])));
  ROOT::Minuit2::FunctionMinimum m = migrad (Rcpp::as<int>(control["maxit"]));
  ROOT::Minuit2::MnHesse() (f, m);

  return pack_results (m);
}

// [[Rcpp::export]]
Rcpp::List migrad_grad_cpp (Rcpp::NumericVector par, Rcpp::Function f_R, Rcpp::Function g_R, Rcpp::NumericVector lower, Rcpp::NumericVector upper, Rcpp::List control) {
  ROOT::Minuit2::MnUserParameters p = pack_parameters(par, lower, upper);

  fcn_grad f(f_R, g_R, Rcpp::as<bool>(control["check.gradient"]));

  ROOT::Minuit2::MnMigrad migrad(f, p, ROOT::Minuit2::MnStrategy(Rcpp::as<int>(control["strategy"])));
  ROOT::Minuit2::FunctionMinimum m = migrad (Rcpp::as<int>(control["maxit"]));
  ROOT::Minuit2::MnHesse() (f, m);

  return pack_results (m);
}

