#include "ceres/ceres.h"
#include "glog/logging.h"
#include <fstream>

typedef Eigen::NumTraits<double> EigenDouble;
typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;
typedef Eigen::Matrix<double, 3, 3> Mat3;
typedef Eigen::Matrix<double, 2, 1> Vec2;
typedef Eigen::Matrix<double, Eigen::Dynamic, 8> MatX8;
typedef Eigen::Vector3d Vec3;

namespace {

/*
  This structure contains options that controls how the homography estimation operates.
*/
struct EstimateHomographyOptions {

  EstimateHomographyOptions(): max_num_iterations(50), expected_average_symmetric_distance(1e-16) {}

  // Maximal number of iterations for the refinement step.
  int max_num_iterations;

  // Expected average of symmetric geometric distance between actual destination points and original ones transformed by estimated homography matrix.
  // Refinement will finish as soon as average of symmetric geometric distance is less or equal to this value.
  // This distance is measured in the same units as input points are.
  double expected_average_symmetric_distance;
};


/*
  Calculate symmetric geometric cost terms:

  forward_error = D(H * x1, x2)
  backward_error = D(H^-1 * x2, x1)

  Templated to be used with autodifferentiation.
*/
template <typename T>

void SymmetricGeometricDistanceTerms(const Eigen::Matrix<T, 3, 3>& H,
                                     const Eigen::Matrix<T, 2, 1>& x1,
                                     const Eigen::Matrix<T, 2, 1>& x2,
                                     T forward_error[2],
                                     T backward_error[2]) {
  typedef Eigen::Matrix<T, 3, 1> Vec3;
  Vec3 x(x1(0), x1(1), T(1.0));
  Vec3 y(x2(0), x2(1), T(1.0));
  Vec3 H_x = H * x;
  Vec3 Hinv_y = H.inverse() * y;

  H_x /= H_x(2);
  Hinv_y /= Hinv_y(2);

  forward_error[0] = H_x(0) - y(0);
  forward_error[1] = H_x(1) - y(1);
  backward_error[0] = Hinv_y(0) - x(0);
  backward_error[1] = Hinv_y(1) - x(1);
}



/*
  Calculate symmetric geometric cost:

  D(H * x1, x2)^2 + D(H^-1 * x2, x1)^2
*/
double SymmetricGeometricDistance(const Mat3& H,
                                  const Vec2& x1,
                                  const Vec2& x2) {
  Vec2 forward_error, backward_error;
  SymmetricGeometricDistanceTerms<double>(H, x1, x2, forward_error.data(), backward_error.data());
  return forward_error.squaredNorm() + backward_error.squaredNorm();
}



/*
  A parameterization of the 2D homography matrix that uses 8 parameters so that the matrix is normalized (H(2,2) == 1).
  The homography matrix H is built from a list of 8 parameters (a, b,...g, h) as follows

           |a b c|
       H = |d e f|
           |g h 1|
*/
template <typename T = double>

class Homography2DNormalizedParameterization {
 public:
  typedef Eigen::Matrix<T, 8, 1> Parameters;     // a, b, ... g, h
  typedef Eigen::Matrix<T, 3, 3> Parameterized;  // H

  // Convert from the 8 parameters to a H matrix.
  static void To(const Parameters& p, Parameterized* h) {
    // clang-format off
    *h << p(0), p(1), p(2),
          p(3), p(4), p(5),
          p(6), p(7), 1.0;
    // clang-format on
  }

  // Convert from a H matrix to the 8 parameters.
  static void From(const Parameterized& h, Parameters* p) {
    // clang-format off
    *p << h(0, 0), h(0, 1), h(0, 2),
          h(1, 0), h(1, 1), h(1, 2),
          h(2, 0), h(2, 1);
    // clang-format on
  }
};



/*
  2D Homography transformation estimation in the case that points are in euclidean coordinates.

  x = H y

  x and y vector must have the same direction, we could write

  crossproduct(|x|, * H * |y| ) = |0|

  | 0 -1  x2|   |a b c|   |y1|    |0|
  | 1  0 -x1| * |d e f| * |y2| =  |0|
  |-x2  x1 0|   |g h 1|   |1 |    |0|

  That gives:

  (-d+x2*g)*y1    + (-e+x2*h)*y2 + -f+x2          |0|
  (a-x1*g)*y1     + (b-x1*h)*y2  + c-x1         = |0|
  (-x2*a+x1*d)*y1 + (-x2*b+x1*e)*y2 + -x2*c+x1*f  |0|
*/
bool Homography2DFromCorrespondencesLinearEuc(const Mat& x1,
                                              const Mat& x2,
                                              Mat3* H,
                                              double expected_precision) {
  assert(2 == x1.rows());
  assert(4 <= x1.cols());
  assert(x1.rows() == x2.rows());
  assert(x1.cols() == x2.cols());

  int n = x1.cols();
  MatX8 L = Mat::Zero(n * 3, 8);
  Mat b = Mat::Zero(n * 3, 1);

  for (int i = 0; i < n; ++i) {
    int j = 3 * i;
    L(j, 0) = x1(0, i);              // a
    L(j, 1) = x1(1, i);              // b
    L(j, 2) = 1.0;                   // c
    L(j, 6) = -x2(0, i) * x1(0, i);  // g
    L(j, 7) = -x2(0, i) * x1(1, i);  // h
    b(j, 0) = x2(0, i);              // i

    ++j;
    L(j, 3) = x1(0, i);              // d
    L(j, 4) = x1(1, i);              // e
    L(j, 5) = 1.0;                   // f
    L(j, 6) = -x2(1, i) * x1(0, i);  // g
    L(j, 7) = -x2(1, i) * x1(1, i);  // h
    b(j, 0) = x2(1, i);              // i

    ++j;
    L(j, 0) = x2(1, i) * x1(0, i);   // a
    L(j, 1) = x2(1, i) * x1(1, i);   // b
    L(j, 2) = x2(1, i);              // c
    L(j, 3) = -x2(0, i) * x1(0, i);  // d
    L(j, 4) = -x2(0, i) * x1(1, i);  // e
    L(j, 5) = -x2(0, i);             // f
  }

  // Solve Lx=b
  const Vec h = L.fullPivLu().solve(b);
  Homography2DNormalizedParameterization<double>::To(h, H);
  return (L * h).isApprox(b, expected_precision);
}



/*
  Cost functor which computes symmetric geometric distance used for homography matrix refinement.
*/
class HomographySymmetricGeometricCostFunctor {

 public:
  HomographySymmetricGeometricCostFunctor(const Vec2& x, const Vec2& y): x_(x), y_(y) {}

  template <typename T>
  bool operator()(const T* homography_parameters, T* residuals) const {
    typedef Eigen::Matrix<T, 3, 3> Mat3;
    typedef Eigen::Matrix<T, 2, 1> Vec2;

    Mat3 H(homography_parameters);
    Vec2 x(T(x_(0)), T(x_(1)));
    Vec2 y(T(y_(0)), T(y_(1)));

    SymmetricGeometricDistanceTerms<T>(H, x, y, &residuals[0], &residuals[2]);
    return true;
  }

  const Vec2 x_;
  const Vec2 y_;
};



/*
  Termination checking callback. This is needed to finish the optimization when an
  absolute error threshold is met, as opposed to Ceres's function_tolerance, which
  provides for finishing when successful steps reduce the cost function by a
  fractional amount.
  In this case, the callback checks for the absolute average reprojection error
  and terminates when it's below a threshold (for example all points < 0.5px error).
*/
class TerminationCheckingCallback : public ceres::IterationCallback {

 public:
  TerminationCheckingCallback(const Mat& x1,
                              const Mat& x2,
                              const EstimateHomographyOptions& options,
                              Mat3* H)
      : options_(options), x1_(x1), x2_(x2), H_(H) {}

  virtual ceres::CallbackReturnType operator()(

      const ceres::IterationSummary& summary) {
    // If the step wasn't successful, there's nothing to do.
    if (!summary.step_is_successful) {
      return ceres::SOLVER_CONTINUE;
    }

    // Calculate average of symmetric geometric distance.
    double average_distance = 0.0;
    for (int i = 0; i < x1_.cols(); i++) {
      average_distance +=
          SymmetricGeometricDistance(*H_, x1_.col(i), x2_.col(i));
    }
    average_distance /= x1_.cols();

    if (average_distance <= options_.expected_average_symmetric_distance) {
      return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
    }

    return ceres::SOLVER_CONTINUE;
  }

 private:
  const EstimateHomographyOptions& options_;
  const Mat& x1_;
  const Mat& x2_;
  Mat3* H_;
};



bool EstimateHomography2DFromCorrespondences(const Mat& x1, const Mat& x2,
                      const EstimateHomographyOptions& options, Mat3* H) {
  assert(2 == x1.rows());
  assert(4 <= x1.cols());
  assert(x1.rows() == x2.rows());
  assert(x1.cols() == x2.cols());

  // Step 1: Algebraic homography estimation.
  // Assume algebraic estimation always succeeds.
  Homography2DFromCorrespondencesLinearEuc(x1, x2, H, EigenDouble::dummy_precision());
  LOG(INFO) << "Estimated matrix after algebraic estimation:\n" << *H;

  // Step 2: Refine matrix using Ceres minimizer.
  ceres::Problem problem;

  for (int i = 0; i < x1.cols(); i++) {
    HomographySymmetricGeometricCostFunctor*
    homography_symmetric_geometric_cost_function =
        new HomographySymmetricGeometricCostFunctor(x1.col(i), x2.col(i));

    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<HomographySymmetricGeometricCostFunctor,
        4,  // num_residuals
        9>(homography_symmetric_geometric_cost_function),
           NULL,
           H->data());
  }

  // Configure the solve.
  ceres::Solver::Options solver_options;
  solver_options.linear_solver_type = ceres::DENSE_QR;
  solver_options.max_num_iterations = options.max_num_iterations;
  solver_options.update_state_every_iteration = true;

  // Terminate if the average symmetric distance is good enough.
  TerminationCheckingCallback callback(x1, x2, options, H);
  solver_options.callbacks.push_back(&callback);

  // Run the solve.
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);
  LOG(INFO) << "Summary:\n" << summary.FullReport();
  LOG(INFO) << "Final refined matrix:\n" << *H;

  return summary.IsSolutionUsable();
}

}  // namespace

using namespace std;

Mat initialise_input_matrix(string arr, int num) {
    string s = ", ";
    Mat mat(2, num);
    int alternate = 0;

    for (int i = 0; i < 2 * num; i++) {
        int index = arr.find(s); // index of ", "
        int mat_idx = int(floor(i / 2.0)); // column for which the coordinate should go to

        if (index < 0) {
            string y = arr.substr(0, arr.length() - 1);
            mat(1, mat_idx) = stod(y);

        } else {
            if (alternate == 0) {
                string x = arr.substr(1, index - 1);
                mat(0,mat_idx) = stod(x);
                alternate = 1;
            } else {
                string y = arr.substr(0, index - 1);
                mat(1, mat_idx) = stod(y);
                alternate = 0;
            }
        }

        arr = arr.substr(index + 2);
    }

    return mat;
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  // Test values for input:
  // [(1, 2), (3, 4), (5, 11), (23, 6)] [(7, 8), (5, 9), (10, 13), (23, 8)]
  // [(13, 15), (0.3, 0.999), (5, 0.0004), (3.2, 1.8)] [(7, 8), (5, 9), (10, 13), (4.5, 2.7)]
  // For magni_chart and mir_5cm: [(480, 231), (228, 219), (549, 472), (205, 461)] [(476, 710), (960, 710), (332, 240), (984, 248)]
  // For CRH and CRH_new: [(199, 135), (1118, 189), (331, 448), (1093, 509)] [(49, 83), (1116, 81), (220, 446), (1124, 460)]
  string input;
  getline(cin, input);
  string delimiter = "]";
  int pos = input.find(delimiter);
  string arr1 = input.substr(1, pos - 1);
  string temp = input.substr(pos + 3);
  string arr2 = temp.substr(0, temp.length() - 1);

  int count = std::count(arr1.begin(), arr1.end(), ',');
  double num_points = ceil(count/2.0);
  Mat x1 = initialise_input_matrix(arr1, num_points);
  Mat x2 = initialise_input_matrix(arr2, num_points);

  Mat3 homography_matrix;
  // This matrix has been dumped from a Blender test file of plane tracking.
  // clang-format off
  homography_matrix << 1.243715, -0.461057, -111.964454,
                       0.0,       0.617589, -192.379252,
                       0.0,      -0.000983,    1.0;
  // clang-format on

  // Mat x1(2, 100);
  // for (int i = 0; i < x1.cols(); ++i) {
  //   x1(0, i) = rand() % 1024;
  //   x1(1, i) = rand() % 1024;
  // }

  // Mat x2 = x1;
  // for (int i = 0; i < x2.cols(); ++i) {
  //   Vec3 homogenous_x1 = Vec3(x1(0, i), x1(1, i), 1.0);
  //   Vec3 homogenous_x2 = homography_matrix * homogenous_x1;

  //   x2(0, i) = homogenous_x2(0) / homogenous_x2(2);
  //   x2(1, i) = homogenous_x2(1) / homogenous_x2(2);

  //   // Apply some noise
  //   x2(0, i) += static_cast<double>(rand() % 1000) / 5000.0;
  //   x2(1, i) += static_cast<double>(rand() % 1000) / 5000.0;
  // }

  std::cout << "Matrix 1:\n" << x1 << std::endl;
  std::cout << "Matrix 2:\n" << x2 << std::endl;

  Mat3 estimated_matrix;
  EstimateHomographyOptions options;
  options.expected_average_symmetric_distance = 0.02;

  EstimateHomography2DFromCorrespondences(x1, x2, options, &estimated_matrix);
  estimated_matrix /= estimated_matrix(2, 2); // Normalize the matrix for easier comparison.

  // std::cout << "Original matrix:\n" << homography_matrix << "\n";
  std::cout << "Estimated matrix:\n" << estimated_matrix << "\n";

  ofstream file;
  file.open("data.txt");
  // file << num_points << endl << arr1 << endl << arr2 << endl << estimated_matrix << endl;
  file << estimated_matrix << endl;
  file.close();

  return EXIT_SUCCESS;
}
