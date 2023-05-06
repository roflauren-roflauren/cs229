import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    gda = GDA()
    gda.fit(x_train, y_train)
    
    # Plot decision boundary on validation set
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    util.plot(x_valid, y_valid, gda.theta, save_path.replace('.txt', '.png'))
    
    # Use np.savetxt to save outputs from validation set to save_path
    preds = gda.predict(x_valid)
    np.savetxt(save_path, preds)
    # *** END CODE HERE ***

class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=1, max_iter=10000, eps=1e-5,
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
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***   
        # Find phi, mu_0, mu_1, and sigma
        phi   = np.count_nonzero(y == 1) / len(y)
        mu_0  = np.expand_dims(
            (np.sum(x[y == 0], axis=0) / np.count_nonzero(y == 0)), axis=-1
        )
        mu_1  = np.expand_dims(
            (np.sum(x[y == 1], axis=0) / np.count_nonzero(y == 1)), axis=-1
        )
        sigma = np.sum([ 
                    (np.expand_dims(x[idx], -1) - mu_0) @ (np.expand_dims(x[idx], -1) - mu_0).T if y[idx] == 0 else 
                    (np.expand_dims(x[idx], -1) - mu_1) @ (np.expand_dims(x[idx], -1) - mu_1).T for idx in range(len(y))
                ], axis=0) / len(y)
    
        # Write theta in terms of the parameters        
        self.theta = np.linalg.inv(sigma) @ (mu_1 - mu_0)
        theta_zero = (mu_0.T @ np.linalg.inv(sigma) @ mu_0) / 2  \
            - (mu_1.T @ np.linalg.inv(sigma) @ mu_1) / 2 - np.log((1 - phi) / phi)
        self.theta = np.concatenate((theta_zero, self.theta))
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        y_hat = 1 / (1 + np.exp(-(x @ self.theta)))
        preds = [1 if elem >= 0.5 else 0 for elem in y_hat]
        return np.asarray(preds)
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')