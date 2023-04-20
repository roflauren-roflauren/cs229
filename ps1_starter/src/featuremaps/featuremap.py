import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')

factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.theta = np.linalg.solve((X.T @ X), (X.T @ y))
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        # temporarily transpose X for easy feature appending: 
        ret = X.T
        # grab the raw X-values for exponentiating: 
        x_raw = ret[1]
        # generate features and append to X matrix: 
        for pow in range(2, k+1): 
           x_powfeature = np.expand_dims(np.array([val ** pow for val in x_raw]), 0)
           ret = np.append(ret, x_powfeature, axis=0)
        # re-transpose and return featurized X: 
        return ret.T 
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        # generate the polynomial features first: 
        ret = self.create_poly(k, X) 
        # temporarily transpose 'ret' and grab raw X-values for np.sin() application:  
        ret = ret.T
        x_raw = ret[1]
        # generate sin() features and append to X matrix: 
        x_sinfeature = np.expand_dims(np.sin(x_raw), 0)
        ret = np.append(ret, x_sinfeature, axis=0)
        # re-transpose and return featurized X: 
        return ret.T
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***       
        return (X @ self.theta)
        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x, train_y = util.load_dataset(train_path, add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        # instantiate the model:
        model = LinearModel()
        # create the feature matrix: 
        x_feats = model.create_poly(k=k, X=train_x) if sine == False else model.create_sin(k=k, X=train_x)        
        # train the model: 
        model.fit(x_feats, train_y)
        # generate predictions using the fitted model parameters: 
        plot_x_feats = model.create_poly(k=k, X=plot_x) if sine == False else model.create_sin(k=k, X=plot_x)
        plot_y = model.predict(plot_x_feats)
        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)

    plt.legend()
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all expetriments
    '''
    # *** START CODE HERE ***
    # spec. for q3b:
    run_exp(train_path, False, [3], 'plot_q3b.png')
    # spec. for q3c: 
    run_exp(train_path, False, [3, 5, 10, 20], 'plot_q3c.png')
    # spec. for q3d: 
    run_exp(train_path, True, [0, 1, 2, 3, 5, 10, 20], 'plot_q3d.png')
    # spec. for q3e: 
    run_exp(small_path, False, [1, 2, 5, 10, 20], 'plot_q3e.png')
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')