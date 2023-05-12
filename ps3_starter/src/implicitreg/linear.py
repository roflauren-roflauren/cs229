import numpy as np
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import util

def generate_plot(betas, X, Y, X_val, Y_val, save_path):
    """Generate a scatter plot of validation error vs. norm

    Args:
        betas: list of numpy arrays, indicating different solutions
        X, Y: training dataset 
        X_val, Y_val: validataion dataset
        save_path: path to save the plot
    """
    # check if the validationerror is zero
    for b in betas:
        assert(np.allclose(X.dot(b), Y))

    # compute the norm and the validation error of all the solutions in list beta
    val_err = []
    norms = []
    for i in range(len(betas)):
        val_err.append(np.mean((X_val.dot(betas[i]) - Y_val) ** 2))
        norms.append(np.linalg.norm(betas[i]))

    # plot the validation error against norm of the solution
    util.plot_points(norms, val_err, save_path)

def linear_model_main():
    save_path_linear = "implicitreg_linear"
    
    train_path = 'ds1_train.csv'
    valid_path = 'ds1_valid.csv'
    X, Y = util.load_dataset(train_path)
    X_val, Y_val = util.load_dataset(valid_path)
    
    beta_0 = None
    # *** START CODE HERE ***
    # find the min norm solution of the training dataset
    # store the results to beta_0
    # *** END CODE HERE ***
    
    assert(np.allclose(X.dot(beta_0), Y))
    
    # ns[i] is orthogonal to all the inputs in the training dataset
    # to help you understand the starter code, check the dimension 
    # of ns before you use it
    ns = null_space(X).T

    # *** START CODE HERE ***
    # find 3 different solutions and generate a scatter plot
    # your plot should include the min norm solution and 3 different solutions
    # you can use the function generate_plot()
    # *** END CODE HERE ***

if __name__ == '__main__':
    linear_model_main()
