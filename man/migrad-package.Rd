\name{migrad-package}
\alias{migrad-package}
\alias{migrad}
\docType{package}
\title{ MIGRAD optimizer from MINUIT }
\description{
  MINUIT is the well known optimization package used in high energy
  physics. This package provides an interface to MIGRAD with an optim-like
  interface.
}

\usage{
optim(par, fn, gr = NULL, \dots,
      method = NULL,
      lower = -Inf, upper = Inf,
      control = list(), hessian = FALSE)

}
\arguments{
 \item{par}{Initial values for the parameters to be optimized over.}
 \item{fn}{A function to be minimized (or maximized), with first
   argument the vector of parameters over which minimization is to take
   place.  It should return a scalar result.}
 \item{gr}{A function to return the gradient; if it is \code{NULL}, a
   finite-difference approximation will be used.}
 \item{\dots}{Further arguments to be passed to \code{fn} and \code{gr}.}
 \item{method}{Ignored, present only for compatibility.}
 \item{lower, upper}{Bounds on the variables.}
 \item{control}{A list of control parameters. See \sQuote{Details}.}
 \item{hessian}{Logical. Should a numerically differentiated Hessian
   matrix be returned?}
}
\details{
  Note that arguments after \code{\dots} must be matched exactly.

  By default \code{optim} performs minimization, but it will maximize
  if \code{control$fnscale} is negative. 

  MINUIT was initially written in Fortran around 1975-1980 at CERN by Fred James.
  Its main field of usage is statistical data analysis of experimental data recorded
  at CERN, but it is also used by people doing data analysis outside CERN or outside
  high energy physics. In 2002 Fred James started a project aiming to re-implement
  MINUIT in an object-oriented way using C++. More information about recent developments,
  releases and installation can be obtained from the MINUIT homepage.
  MIGRAD is a variable metric minimizer available in the MINUIT package.

  Function \code{fn} should always return finite values.

  \code{migrad} can be used recursively, and for a single parameter
  as well as many.  It also accepts a zero-length \code{par}, and just
  evaluates the function with that argument.

  The \code{control} argument is a list that can supply any of the
  following components:
  \describe{
    \item{\code{trace}}{Non-negative integer. If positive,
      tracing information on the progress of the optimization 
      is produced.}
    \item{\code{fnscale}}{An overall scaling to be applied to the value
      of \code{fn} and \code{gr} during optimization. If negative,
      turns the problem into a maximization problem. Optimization is
      performed on \code{fn(par)/fnscale}.}
    \item{\code{maxit}}{The maximum number of iterations. Defaults is 0, which
      means migrad will choose by itself.}
    \item{\code{check.gradient}}{\code{migrad} provide an internal consistency
      check of the gradient (if provided): in case the check fails the program
      is aborted. To prevent this, set this flag to \code{FALSE}.}
    \item{\code{strategy}}{Defines the accuracy used during the minimization:
      accepted values \code{0} (lower accuracy) to \code{2} (longer run). 
      Default is \code{1}.}
  }

  Any names given to \code{par} will be copied to the vectors passed to
  \code{fn} and \code{gr}.  Note that no other attributes of \code{par}
  are copied over.
}
%% when numerical derivatives are used, fn is called repeatedly with
%% modified copies of the same objet.

\value{
  For \code{migrad}, a list with components:
  \item{par}{The best set of parameters found.}
  \item{value}{The value of \code{fn} corresponding to \code{par}.}
  \item{counts}{A two-element integer vector giving the number of calls
    to \code{fn} and \code{gr} respectively. This excludes those calls needed
    to compute the Hessian, if requested, and any calls to \code{fn} to
    compute a finite-difference approximation to the gradient.}
  \item{convergence}{An integer code. \code{0} indicates successful
    completion.}
  \item{message}{A character string giving any additional information
    returned by the optimizer, or \code{NULL}.}
  \item{hessian}{Only if argument \code{hessian} is true. A symmetric
    matrix giving an estimate of the Hessian at the solution found.  Note
    that this is the Hessian of the unconstrained problem even if the
    box constraints are active.}
}

\references{
  CN/ASD Group. MINUIT - Users Guide, nProgram Library D506. CERN, 1993

  MINUIT/MIGRAD home page \url{http://www.cern.ch/minuit}
}
\keyword{optimizer}
\seealso{optim}
\examples{
  # Optional simple examples of the most important functions
}


\keyword{nonlinear}
\keyword{optimize}
