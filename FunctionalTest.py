import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy import stats,random
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
import seaborn as sns


#This script is a self-contained test demonstrating how to calculate
# p-values associated with differences between two arbitrary traces
# fit with splines.

# !!!!! I don't yet consider this test to be complete


def GenerateCholesky(matrixSize=7):

    #Generate a positive semi-definite 'covariance' matrix
    # (generate a random matrix, multiply it by its own transpose)

    A = random.rand(matrixSize,matrixSize)
    B = np.dot(A,A.transpose())

    #Convert this random 'covariance' matrix into a correlation matrix 
    B = np.asanyarray(B)
    std = np.sqrt(np.diag(B))

    corr = B / np.outer(std,std)

    print 'random positive semi-define matrix for today is\n', corr

    #Calculate the Cholesky decomposition so we can throw
    # random vectors with correlations
    c = cholesky(corr,lower=True)

    return c

def DegreesOfFreedom(c,**kwargs):
    #Use Cholesky decomposition to simulate from the covariance matrix,
    # fit a chi-squared function to the simulated residuals to 
    # extract an effective degrees of freedom

    #Make a set of correlated random vectors

    num_samples = kwargs.get("num_samples",500)
    matrixSize = c.shape[0]

    # Generate samples from independent normally distributed random
    # variables (with mean 0 and std. dev. 1).
    x1 = stats.norm.rvs(size=(matrixSize,num_samples)) 
    x2 = stats.norm.rvs(size=(matrixSize,num_samples)) 

    #Simluate lots of correlated draws, calculate & plot the p-value
    res_ab_dist = []
    res_a_dist = []

    for i in range(50000):
        x1 = stats.norm.rvs(size=(matrixSize))
        x2 = stats.norm.rvs(size=(matrixSize))
        
        y1 = np.dot(c, x1)
        y2 = np.dot(c, x2)
        
        x_vec = range(matrixSize)

        res_ab,res_a,res_b = Functional_pval(a_vec   = y1,
                                        b_vec   = y2,
                                        a_xvals = x_vec,
                                        b_xvals = x_vec)

        res_ab_dist.append(res_ab)
        res_a_dist.append(res_a + res_b)
        
        
    #Fit a chi-square function to these distributions to
    #determine the degrees of freedom

    #This function returns the degrees-of-freedom, location(mean), scale
    fit1 = stats.chi2.fit(res_ab_dist)
    fit2 = stats.chi2.fit(res_a_dist)
    df1 = fit1[0]
    df2 = fit2[0]

    if kwargs.get("plot",False):
    #Draw these distributions & fits for future reference
        x = np.linspace(stats.chi2.ppf(0.01, df1),stats.chi2.ppf(0.99, df1), 100)
        plt.hist(res_ab_dist,alpha=0.5,bins=x,normed=True,label="H0",color='b')
        plt.plot(x,stats.chi2.pdf(x,*fit1),color='b')
        plt.hist(res_a_dist,alpha=0.5,bins=x,normed=True,label="H1",color='r')
        plt.plot(x,stats.chi2.pdf(x,*fit2),color='r')
        plt.legend(loc='upper right',fontsize=15)
        plt.xlabel("Residual",fontsize=20)
        plt.ylabel("Frequency (normalized)",fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.show()
        
    return df1,df2



def Functional_pval(a_vec,b_vec,a_xvals,b_xvals,d1 = None, d2 = None):
    #Use spline fitting & and functional analysis to calculate a 
    # similarity p-value.

    #First, we calculate a cubic spline for each trend, 
    # which stores the residual in the returned object
    f_a = UnivariateSpline(x = a_xvals,
                           y = a_vec)
    
    f_b = UnivariateSpline(x = b_xvals,
                           y = b_vec)

    #Next, we calculate the cubic spline of the null hypothesis, 
    # assuming both measured trends are drawn from the same 
    # underlying smooth trend. This requires some book-keeping.

    #x values need to be strictly increasing, but we have multiple
    # values for some x's in a and b.  So let's offset by an infintesimal
    # amount whenever there's a repeat. 
    b_xvals = [x + 0.00001 if x in a_xvals else x for x in b_xvals]

    #So when we join our two trends together, the xvalues sort nicely
    full_range = sorted(list(a_xvals)+list(b_xvals))

    #BUT we have to keep track of the indexing too, 
    # for when we join the a and b y-value lists together.
    # This ID's the index order we want to draw from the joined list
    new_indexes = [i[0] for i in sorted(enumerate(list(a_xvals)+list(b_xvals)),key=lambda x: x[1])]

    #Combine the y-value lists and order them to match the x-values
    full_vec = list(a_vec) + list(b_vec)
    result = [full_vec[x] for x in new_indexes]

    #Fit the null hypothesis spline
    f_ab = UnivariateSpline(x = full_range,
                            y = result)
    
    #these are already the sum of squared resisuals
    res_a = f_a.get_residual()
    res_b = f_b.get_residual()
    res_ab = f_ab.get_residual()

    
    #On the first pass, we're just using this function to 
    # derive the distributions of residuals. Don't calculate
    # the F-score, just return the residuals
    if d1 == None and d2 == None:
        return res_ab,res_a,res_b
        
    else:

        #F-test is  F = (d2/d1) * (RSS0-RSS1)/(RSS1)
        # where RSS is residual sum of squares,
        # RSS0 is the null hypothesis residual,
        # RSS1 is the sum of residuals from separate models (res_a+res_b),
        # and d1 and d2 are the degrees of freedom. 
        
        # Comparing 2 groups, df_Numerator = k - 1 = 2 - 1
        #d2 = 1.0
        
        #df_Denominator = N - k = N - 1 = 
        #d1 = len(result) - 1.0
        
        try:
            F = (d2/d1) * (res_ab - (res_a + res_b)) / (res_a + res_b)
        except:
            #Add a little residual if the cubic spline fit is _too_ good
            # (a divide-by-zero error)
            F = (d2/d1) * (res_ab - (res_a + res_b)) / (res_a + res_b + 0.001)

        #The p-value here is the probability that two trends, drawn
        # from the same underlying smooth trend, would give a better
        # RSS from separate splines rather than the same spline
        p_val = stats.f.cdf(F,d2,d1)
        
        return p_val

def FunctionalUnitTest(**kwargs):


    #Let's generate a fake dataset of random vectors,
    # some with correlations (representing H0, their fits should
    # be described by the same underlying smooth trend)
    
    #Make a set of correlated random vectors
    num_samples = kwargs.get("num_samples",500)
    matrixSize = kwargs.get("vector_size",7)

    c = GenerateCholesky(matrixSize)
    df1, df2 = DegreesOfFreedom(c,plot=True)

    # Generate samples from independent normally distributed random
    # variables (with mean 0 and std. dev. 1).
    x1 = stats.norm.rvs(size=(matrixSize,num_samples)) 
    x2 = stats.norm.rvs(size=(matrixSize,num_samples)) 

    #Make two datasets, one correlated and one uncorrelated (random)
    y1 = np.dot(c, x1) #Correlated
    y2 = x2 #Random
    
    #
    # Plot various projections of the samples.
    #
    if kwargs.get("plot",False):
        plt.plot(range(matrixSize),np.diagonal(c))
        plt.xlabel("Diagonal Element",fontsize=20)
        plt.ylabel("Covariance Magnitude",fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.show()

        plt.subplot(2,2,1)
        plt.plot(y1[0], y1[1], 'b.',label="Correlated")
        plt.plot(y2[0], y2[1], 'r.',label="Un-correlated")
        plt.ylabel('Element 1')
        plt.xlabel('Element 0')
        plt.axis('equal')
        plt.legend(loc='upper left')
        plt.grid(True)

        plt.subplot(2,2,2)
        plt.plot(y1[0], y1[2], 'b.')
        plt.plot(y2[0], y2[2], 'r.')
        plt.xlabel('Element 0')
        plt.ylabel('Element 2')
        plt.axis('equal')
        plt.grid(True)
        
        plt.subplot(2,2,3)
        plt.plot(y1[1], y1[2], 'b.')
        plt.plot(y2[1], y2[2], 'r.')
        plt.xlabel('Element 1')
        plt.ylabel('Element 2')
        plt.axis('equal')
        plt.grid(True)
        
        plt.subplot(2,2,4)
        plt.plot(y1[2], y1[3], 'b.')
        plt.plot(y2[2], y2[3], 'r.')
        plt.xlabel('Element 2')
        plt.ylabel('Element 3')
        plt.axis('equal')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    x1 = stats.norm.rvs(size=(matrixSize))
    x2 = stats.norm.rvs(size=(matrixSize))
    
    y1 = np.dot(c, x1)
    y2 = np.dot(c, x2)
    
    x_vec = range(matrixSize)

    print "This should reflect that the trends are similar, so the"
    print "null hypothesis is favored with a high p-value:"
    print Functional_pval(a_vec   = y1,
                     b_vec   = y2,
                     a_xvals = x_vec,
                     b_xvals = x_vec,
                     d1 = df1,
                     d2 = df2)
    print "\n"

    print "This should reflect that the trends are different, so the"
    print "null hypothesis is dis-favored with a low p-value:"
    print Functional_pval(a_vec   = y1,
                     b_vec   = x2,
                     a_xvals = x_vec,
                     b_xvals = x_vec,
                     d1 = df1,
                     d2 = df2)
    print "\n"





    #Simluate lots of correlated draws, calculate & plot the p-value
    p_vals = []
    p_vals2 = []
    for i in range(50000):
        x1 = stats.norm.rvs(size=(matrixSize))
        x2 = stats.norm.rvs(size=(matrixSize))
        
        y1 = np.dot(c, x1)
        y2 = np.dot(c, x2)
        
        x_vec = range(matrixSize)

        x_vec_fine = np.linspace(0,matrixSize-1,100) 

        p_vals.append(Functional_pval(a_vec   = y1,
                                 b_vec   = y2,
                                 a_xvals = x_vec,
                                 b_xvals = x_vec,
                                 d1 = df1,
                                 d2 = df2))

        #Let's see some examples of correlated traces
        if i<3:

            plt.plot(x_vec, y1, color ='b')
            plt.plot([x + 0.1 for x in x_vec], y2, color='r')

            f_a = UnivariateSpline(x = x_vec,
                                   y = y1)
            
            f_b = UnivariateSpline(x = x_vec,
                                   y = y2)

            plt.plot(x_vec_fine,f_a(x_vec_fine),color='b',alpha=0.5)
            plt.plot(x_vec_fine,f_b(x_vec_fine),color='r',alpha=0.5)

            plt.xlabel("X series",fontsize=15)
            plt.title("Example "+str(i+1)+"/3",fontsize=15)
            plt.tight_layout()
            plt.show()


        #Now compare two traces that are uncorrelated 
        # (one contains correlations from the matrix, one is 
        # strictly random)
        x1 = stats.norm.rvs(size=(matrixSize))
        x2 = stats.norm.rvs(size=(matrixSize))
        y1 = np.dot(c, x1)
        
        p_vals2.append(Functional_pval(a_vec  = y1,
                                  b_vec   = x2,
                                  a_xvals = x_vec,
                                  b_xvals = x_vec,
                                  d1 = df1,
                                  d2 = df2))

       
    plt.subplot(211)
    plt.hist(p_vals,bins=np.linspace(0,1,100),histtype='step',linewidth=2,normed=True,label = "Correlated")
    plt.hist(p_vals2,bins=np.linspace(0,1,100),histtype='step',linewidth=2,normed=True,label = "Un-correlated")
    plt.axvline(0.05,color='gray')

    plt.xlabel("p-value",fontsize=15)
    plt.ylabel("Frequency (normalized)",fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='lower right')

    plt.subplot(212)
    plt.plot(range(10000),sorted(p_vals[:10000]))
    plt.plot(range(10000),sorted(p_vals2[:10000]))
    plt.semilogy()
    plt.ylabel("p-value",fontsize=15)
    plt.xlabel("trace (sorted)",fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.show()
        

def main():

    FunctionalUnitTest(vector_size=6,plot=True)
    FunctionalUnitTest(vector_size=8)
    FunctionalUnitTest(vector_size=10)



if __name__ == "__main__":
    main()
