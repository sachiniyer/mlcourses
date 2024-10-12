#include <cmath>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * Z = X @ theta
 * grad = np.zeros_like(theta)
 * for i in range(X.shape[0] // batch):
 *     X_batch = X[i * batch : (i + 1) * batch]
 *     y_batch = y[i * batch : (i + 1) * batch]
 *     Z_batch = X_batch @ theta
 *     exp_Z = np.exp(Z_batch)
 *     grad = X_batch.T @ (
 *         exp_Z / np.sum(exp_Z, axis=1)[:, None] -
 * np.eye(exp_Z.shape[1])[y_batch]
 *     )
 *     theta -= lr * grad / batch
 */
void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch) {
  /**
   * A C++ version of the softmax regression epoch code.  This should run a
   * single epoch over the data defined by X and y (and sizes m,n,k), and
   * modify theta in place.  Your function will probably want to allocate
   * (and then delete) some helper arrays to store the logits and gradients.
   *
   * Args:
   *     X (const float *): pointer to X data, of size m*n, stored in row
   *          major (C) format
   *     y (const unsigned char *): pointer to y data, of size m
   *     theta (float *): pointer to theta data, of size n*k, stored in row
   *          major (C) format
   *     m (size_t): number of examples
   *     n (size_t): input dimension
   *     k (size_t): number of classes
   *     lr (float): learning rate / SGD step size
   *     batch (int): SGD minibatch size
   *
   * Returns:
   *     (None)
   */

  /// BEGIN YOUR CODE
  float *Z = new float[m * k];
  float *grad = new float[n * k];
  for (size_t i = 0; i < m / batch; i++) {
    const float *X_batch = X + i * batch * n;
    const unsigned char *y_batch = y + i * batch;
    float *Z_batch = Z + i * batch * k;
    for (size_t j = 0; j < batch; j++) {
      for (size_t l = 0; l < k; l++) {
        Z_batch[j * k + l] = 0;
        for (size_t m = 0; m < n; m++) {
          Z_batch[j * k + l] += X_batch[j * n + m] * theta[m * k + l];
        }
      }
    }
    float *exp_Z = new float[batch * k];
    for (size_t j = 0; j < batch; j++) {
      for (size_t l = 0; l < k; l++) {
        exp_Z[j * k + l] = exp(Z_batch[j * k + l]);
      }
    }
    float *sum_exp_Z = new float[batch];
    for (size_t j = 0; j < batch; j++) {
      sum_exp_Z[j] = 0;
      for (size_t l = 0; l < k; l++) {
        sum_exp_Z[j] += exp_Z[j * k + l];
      }
    }
    float *grad_batch = new float[n * k];
    for (size_t j = 0; j < batch; j++) {
      for (size_t l = 0; l < k; l++) {
        grad_batch[j * k + l] = exp_Z[j * k + l] / sum_exp_Z[j];
      }
      grad_batch[j * k + y_batch[j]] -= 1;
    }
    for (size_t j = 0; j < n; j++) {
      for (size_t l = 0; l < k; l++) {
        grad[j * k + l] = 0;
        for (size_t m = 0; m < batch; m++) {
          grad[j * k + l] += X_batch[m * n + j] * grad_batch[m * k + l];
        }
      }
    }
    for (size_t j = 0; j < n; j++) {
      for (size_t l = 0; l < k; l++) {
        theta[j * k + l] -= lr * grad[j * k + l] / batch;
      }
    }
    delete[] exp_Z;
    delete[] sum_exp_Z;
    delete[] grad_batch;
  }
  /// END YOUR CODE
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
  m.def(
      "softmax_regression_epoch_cpp",
      [](py::array_t<float, py::array::c_style> X,
         py::array_t<unsigned char, py::array::c_style> y,
         py::array_t<float, py::array::c_style> theta, float lr, int batch) {
        softmax_regression_epoch_cpp(
            static_cast<const float *>(X.request().ptr),
            static_cast<const unsigned char *>(y.request().ptr),
            static_cast<float *>(theta.request().ptr), X.request().shape[0],
            X.request().shape[1], theta.request().shape[1], lr, batch);
      },
      py::arg("X"), py::arg("y"), py::arg("theta"), py::arg("lr"),
      py::arg("batch"));
}
