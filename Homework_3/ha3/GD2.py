
import numpy as np
# X          - single array/vector
# y          - single array/vector
# theta      - single array/vector
# alpha      - scalar
# iterations - scarlar

def gradientDescent(X, y, theta, alpha, numIterations):
    '''
        # This function returns a tuple (theta, Cost array)
        '''
    m = len(y)
    arrCost =[];
    transposedX = np.transpose(X) # transpose X into a vector  -> XColCount X m matrix
    for interation in range(0, numIterations):
        ################PLACEHOLDER3 #start##########################
        #: write your codes to update theta, i.e., the parameters to estimate.
        hyp =  np.dot(X, theta)
        residualError =  hyp - y
        gradient =  np.dot(transposedX, residualError) / m
        change = [alpha * x for x in gradient]
        theta = np.subtract(theta, change)  # theta = theta - alpha * gradient
        ################PLACEHOLDER3 #end##########################
        
        ################PLACEHOLDER4 #start##########################
        # calculate the current cost with the new theta;
        atmp =  np.sum(residualError**2) / (m)
    
        print(atmp)
        arrCost.append(atmp)
    # cost = (1 / m) * np.sum(residualError ** 2)
    ################PLACEHOLDER4 #start##########################
    
    return theta, arrCost
