import numpy as np
import matplotlib.pyplot as plt
from random import randint
import imageio
import time
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)

SLOPE_DATA = 2.0
INTERCEPT_DATA = -1.0
NOISE_DATA = 0.2
X_true = [randint(1, 100) * 0.01 for x in range(2000)]
y_true = [SLOPE_DATA * x + INTERCEPT_DATA + np.random.normal(0, NOISE_DATA) for x in X_true]
SLOPE_STARTING = -5.5
INTERCEPT_STARTING = -3.0
LEARNING_RATE = 0.08

def make_line(x_data, slope, intercept):
    return [slope * x + intercept for x in x_data]

def mse(ytrue, ypred):
    return sum([(yt-yp)**2 for yt, yp in zip(ytrue, ypred)]) / len(ytrue)

def gradient_of_loss_function(Xtrue, ytrue, slope, intercept):
    step_size = 0.001

    # we want to make an initial linear regression line, to improve it later
    first_line = make_line(Xtrue, slope, intercept)

    # tweak the slope
    new_slope = slope + step_size
    new_slope_line = make_line(Xtrue, new_slope, intercept)

    # tweak the intercept
    new_intercept = intercept + step_size
    new_intercept_line = make_line(Xtrue, slope, new_intercept)

    # calculate the gradient for those tweaks (change in y over the change in x).
    new_slope_gradient = ((mse(ytrue, new_slope_line) - mse(ytrue, first_line)) / step_size)

    new_intercept_gradient = ((mse(ytrue, new_intercept_line) - mse(ytrue, first_line)) / step_size)

    # return the gradient
    return new_slope_gradient, new_intercept_gradient

def gradient_steps(Xtrue, ytrue, slope, intercept, learning_rate):

    for i in range(500):
        time.sleep(0.01)

        # Get the gradients
        slope_gradient, intercept_gradient = gradient_of_loss_function(Xtrue, ytrue, slope, intercept)

        # Update our slope and intercept based on them
        SLOPE_NEW = slope - slope_gradient * learning_rate
        INTERCEPT_NEW = intercept - intercept_gradient * learning_rate

        # Reset our slope and intercept
        slope = SLOPE_NEW
        intercept = INTERCEPT_NEW

        # Prints only every 10th interation
        if i % 10 == 0:
            plt.scatter(Xtrue, ytrue, s=0.5)
            plt.plot(Xtrue, make_line(Xtrue, slope, intercept))
            plt.title(f'gradient_descent_at_step_{i}')
            plt.savefig(f'plots/step_{i}.png')
            plt.figure()
    return

def create_gif():
    images = []
    for i in range(500):
        if i % 10 == 0:
            filename = f'plots/step_{i}.png'
            image = imageio.imread(filename)
            images.append(image)

    imageio.mimsave('linear_gradient.gif', images, fps=15)
    return imageio.mimsave('linear_gradient.gif', images, fps=15)

gradient_steps(X_true, y_true, SLOPE_STARTING, INTERCEPT_STARTING, LEARNING_RATE)
create_gif()
