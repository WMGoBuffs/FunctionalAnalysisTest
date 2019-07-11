# FunctionalAnalysisTest

This script is an attempt to generalize the Functional Analysis approach to calculating p-values presented in http://dx.doi.org/10.1101/373845.  This paper presents a method using a particular model, but I want to apply it to sparse time series with spline fits.  

The comparison I'm making is between a control time series and a time series with an added perturbation, with somewhere between 4 and 20 sampled measurements through time. The functional analysis approach is to calculate the residual sum of squares (RSS) for different spline fits. First, for the null hypothesis, we assume the perturbation has no effect and we fit a single spline to both traces, and calculate the RSS.  For the alternative hypothesis, we fit two splines and calculate the sum of the two RSS's.  A F-statistic is calculated to discriminate between these two models, which can then be converted into a p-value. 

I don't want to hinge this test on a particlar covariance matrix from my data (i.e. I want it to be general), so I generate a random covariance matrix and use that to 'throw' random vectors with correlations, which should favor the null hypothesis when I perform the comparison (they should have a smaller RSS when fit with a single spline). I also compare a thrown vector to a random vector, which should favor the alternative hypothesis (they should render a small RSS when fit with a single spline).

It's difficult to construct a unit test for something like this, since I don't have a particular 'true' value I'm expecting to compare to the function output. I'm more interested in the plots of distributions at this point.  The FunctionalUnitTest function takes a sparse time series size, generates the covariance matrix, and calculates the effective degrees of freedom represented by the data in the covariance matrix. I then generate 50k correlated pairs of time series and 50k uncorrelated pairs, and perform the p-value calculation. The distributions of p-values of these two classes are plotted, which provide the feedback I'm interested in for validation. I also find it useful to look at the sorted p-values represented in the distribution.
