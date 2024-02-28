"""
Gradient Descent to Minimize Function

Summary:
Implement gradient descent to minimize the function f(x) = x^2.

Functions:
"""

def get_minimizer(iterations: int, learning_rate: float, init: int) -> float:
    """
    Perform gradient descent to minimize the function f(x) = x^2.

    Parameters:
    - iterations (int): Number of iterations for the optimization.
    - learning_rate (float): Learning rate for gradient descent.
    - init (int): Initial guess for the minimization.

    Returns:
    - float: Minimized value of the function.

    """
    # Pre-computed derivative of the provided function
    deriv_func = lambda x: 2.0 * x
    current_guess = init

    for i in range(0, iterations):
        delta = deriv_func(current_guess)
        current_guess -= delta * learning_rate

    return round(current_guess, 5)


# Unit-test 1
iterations, learning_rate, init, expected_res = 10, 0.01, 5, 4.08536

# Value after minimizing the function
min_val = get_minimizer(iterations=iterations, learning_rate=learning_rate, init=init)

# Assert that the values are close within the specified tolerance
assert abs(min_val - expected_res) <= 1e-5, "Values are not close"
