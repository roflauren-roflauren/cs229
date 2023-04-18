import numpy as np
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    model = PoissonRegression(step_size=lr)
    model.fit(x_train, y_train) 
    # Run on the validation set, and use np.savetxt to save outputs to save_path
    x_eval, _ = util.load_dataset(eval_path, add_intercept=True)
    preds = model.predict(x_eval)    
    np.savetxt(fname=save_path, X=preds)
    # *** END CODE HERE ***

class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.
        Update the parameter by step_size * (sum of the gradient over examples)

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***        
        # initialize theta to zero vector if no initial guess is provided:         
        self.theta = np.zeros(shape=(x.shape[1], 1)) if self.theta is None else self.theta
        # unsqueeze y-vector: 
        y = np.expand_dims(y, axis=1)
        # conduct gradient ascent for the specified number of iterations: 
        for idx in range(self.max_iter): 
            # compute the sum of the gradient over examples: 
            batched_grad = np.expand_dims(np.sum(x * (y - np.exp(self.theta.T @ x.T).T), axis=0), axis=1)
            assert(batched_grad.shape == self.theta.shape)
            # update theta (and retain old one for convergence check): 
            theta_old  = self.theta 
            self.theta = self.theta + self.step_size * batched_grad  
            # if verbose, compute and print training loss: 
            if self.verbose:         
                if idx % 10 == 0:                
                    training_loss = 1 / y.shape[0] * np.sum(np.square(y - self.predict(x)))
                    print(f'Training loss for epoch {idx}: {training_loss}')            
            # check for convergence:
            if np.sqrt(np.sum(np.square(self.theta - theta_old))) < self.eps:
                break 
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***        
        return np.exp(x @ self.theta)
        # *** END CODE HERE ***

if __name__ == '__main__':
    # run regression:
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
    
    # code to generate plot of predicted vs true counts: 
    preds_file = open('poisson_pred.txt', 'r')
    preds = preds_file.read().split('\n')
    preds = [float(pred) for pred in preds if pred != ''] 
    _, y_eval = util.load_dataset('valid.csv', add_intercept=True)
    plt.scatter(y_eval, preds)
    plt.xlabel("True Count")
    plt.ylabel("Predicted Expected Count")
    plt.show()