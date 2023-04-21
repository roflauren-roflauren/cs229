import numpy as np
import util
import sys
from random import random

sys.path.append('../linearclass')

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1

def accuracy_report(preds, y_valid):
    num_examples = len(y_valid)
    binned_preds = [1 if pred >= 0.5 else 0 for pred in preds]
    assert (len(binned_preds) == num_examples)
    
    # count correct predictions: 
    overall, neg, pos = 0, 0, 0
    for idx in range(num_examples): 
        if binned_preds[idx] == y_valid[idx]: 
            if binned_preds[idx] == 0: 
                neg += 1
            if binned_preds[idx] == 1: 
                pos += 1
            overall += 1
    
    # compute accuracies: 
    num_neg_examples, num_pos_examples = list(y_valid).count(0), list(y_valid).count(1)
    assert (num_neg_examples + num_pos_examples == num_examples)
    overall_acc, neg_acc, pos_acc = overall/num_examples, neg/num_neg_examples, pos/num_pos_examples
    balanced_acc = 0.5 * (neg_acc + pos_acc)
    
    # accuracies report: 
    print(f'Overall accuracy :: {overall_acc:.2%}\n' + \
          f'Balanced accuracy :: {balanced_acc:.2%}\n' + \
          f'Negative class accuracy :: {neg_acc:.2%}\n' + \
          f'Positive class accuracy :: {pos_acc:.2%}\n')
    
def main(train_path, validation_path, save_path):
    """Problem 2: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')

    # *** START CODE HERE ***
    # Part (b): Vanilla logistic regression
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    
    # load datasets: 
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(validation_path, add_intercept=True)
    
    # instantiate and fit model: 
    model = LogisticRegression()
    model.fit(x_train, y_train)
    
    # generate predictions using model and save: 
    preds = model.predict(x_valid)
    np.savetxt(output_path_naive, preds)
    
    # generate and save plot: 
    util.plot(x_valid, y_valid, model.theta, './plot_q4b.png')
    
    # compute and report overall, per-class, and balanced accuracies: 
    print("\nUSING RAW TRAINING DATA:")
    accuracy_report(preds, y_valid)
    
    # Part (d): Upsampling minority class
    # Make sure to save predicted probabilities to output_path_upsampling using np.savetxt()
    # Repeat minority examples 1 / kappa times
    
    # generate augmentation data with (1 / kappa) - 1 copies of each minority (positive label) example: 
    # NOTE: we subtract 1 because 1 copy of the data point already exists in the training dataset. 
    x_augment, y_augment = [], []

    for x, y in zip(x_train, y_train):
        # if data point corresponds to minority class label: 
        if y == 1: 
            # create the necessary copies and add the values to the augmentation data 
            for _ in range(int((1/kappa) - 1)): 
                x_augment.append(x), y_augment.append(y)
    
    # add the augmentation data to the training data 
    x_train_prime = np.append(x_train, x_augment, axis=0)
    y_train_prime = np.append(y_train, np.array(y_augment), axis=0)

    # instantiate and fit model:
    upsample_model = LogisticRegression()
    upsample_model.fit(x_train_prime, y_train_prime)
    
    # generate predictions using model and save:
    upsample_preds = upsample_model.predict(x_valid)
    np.savetxt(output_path_upsampling, upsample_preds)
    
    # generate and save plot: 
    util.plot(x_valid, y_valid, upsample_model.theta, './plot_q4d.png')
    
    # compute and report overall, pre-class, and balanaced accuracies: 
    print("\nUSING UPSAMPLED TRAINING DATA:")
    accuracy_report(upsample_preds, y_valid)
    # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        validation_path='validation.csv',
        save_path='imbalanced_X_pred.txt')
