# Imports
import numpy as np
import scipy.optimize as opt
from Q2a import gradient_loglikelihood, viterbi_decode_WTX

def compare(predictions, true_labels):
    """Compare the predicted labels with the true labels and compute the accuracy"""
    correct = 0
    total = 0
    for pred, true in zip(predictions, true_labels):
        for p, t in zip(pred, true):
            if p == t:
                correct += 1
            total += 1
    return correct / total

def get_crf_obj(word_list, W, T, c):

    gradW = np.zeros_like(W)
    gradT = np.zeros_like(T)

    for X, y in word_list:
        dW, dT = gradient_loglikelihood(X, y, W, T)  # gradient of +logp
        gradW += dW
        gradT += dT

    # If objective is negative log-likelihood:
    gradW = -gradW + c * W
    gradT = -gradT + c * T

    return gradW, gradT

def crf_obj(x, word_list, c):
    """Compute the CRF objective and gradient on the list of words (word_list)
    evaluated at the current model x (w_y and T, stored as a vector)
    """
    
    # x is a vector as required by the solver. So reshape it to w_y and T
    W = np.reshape(x[128*26], (128, 26))  # each column of W is w_y (128 dim)
    T = np.reshape(x[128*26:], (26, 26))  # T is 26*26

    f = get_crf_obj(word_list, W, T, c)  # Compute the objective value of CRF
                                         # objective log-likelihood + regularizer
    g_W = f[0]                 # compute the gradient in W(128 * 26)
    g_T = f[1]                  # compute the gradient in T(26*26)
    g = np.concatenate([g_W.reshape(-1), g_T.reshape(-1)])  # Flatten the
                                                          # gradient back into
                                                          # a vector
    return [f,g]

def crf_test(x, word_list):
    """
    Compute the test accuracy on the list of words (word_list); x is the
    current model (w_y and T, stored as a vector)
    """

    # x is a vector. so reshape it into w_y and T
    W = np.reshape(x[:128*26], (128, 26))  # each column of W is w_y (128 dim)
    T = np.reshape(x[128*26:], (26, 26))  # T is 26*26

    # Compute the CRF prediction of test data using W and T
    y_predict = viterbi_decode_WTX(W, T, word_list)

    # Compute the test accuracy by comparing the prediction with the ground truth
    true_label_of_word_list = viterbi_decode_WTX(W, T, word_list, return_labels=True)
    accuracy = compare(y_predict, true_label_of_word_list)
    print('Accuracy = {}\n'.format(accuracy))

def ref_optimize(train_data, test_data, c):
    print('Training CRF ... c = {} \n'.format(c))

    # Initial value of the parameters W and T, stored in a vector
    x0 = np.zeros((128*26+26**2,1))

    # Start the optimization
    result = opt.fmin_tnc(crf_obj, x0, args = [train_data, c], maxfun=100,
                          ftol=1e-3, disp=5)
    model  = result[0]          # model is the solution returned by the optimizer
    accuracy = crf_test(model, test_data)
    print('CRF test accuracy for c = {}: {}'.format(c, accuracy))
    return accuracy

def main():
    # Load the data
    train_data = np.load("data/train_data.npy", allow_pickle=True)
    test_data = np.load("data/test_data.npy", allow_pickle=True)

    # Regularization strength
    c = 0.1

    # Train the CRF and evaluate on the test set
    accuracy = ref_optimize(train_data, test_data, c)
    error = 1 - accuracy
    print('CRF test error for c = {}: {}'.format(c, error))




    
    
