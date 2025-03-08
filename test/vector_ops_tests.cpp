// test/vector_ops_tests.cpp
#include "ann/vector_ops.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

// Global file stream
std::ofstream logfile;

namespace {

bool isApproxEqual(float a, float b, float epsilon = 0.0001f) {
  return std::fabs(a - b) < epsilon;
}

bool isApproxEqual(const std::vector<float> &v1, const std::vector<float> &v2,
                   float epsilon = 0.0001f) {
  if (v1.size() != v2.size())
    return false;
  for (size_t i = 0; i < v1.size(); ++i) {
    if (!isApproxEqual(v1[i], v2[i], epsilon))
      return false;
  }
  return true;
}

std::string vecToString(const std::vector<float> &v) {
  std::ostringstream ss;
  ss << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    ss << v[i] << (i < v.size() - 1 ? ", " : "");
  }
  ss << "]";
  return ss.str();
}

void logOutput(const std::string &message) {
  std::cout << message;
  if (logfile.is_open()) {
    logfile << message;
  }
}

bool testResult(const std::string &name, float actual, float expected) {
  bool passed = isApproxEqual(actual, expected);
  std::ostringstream ss;
  ss << std::left << std::setw(40) << name + ":";
  ss << (passed ? "PASSED"
                : "FAILED (Expected: " + std::to_string(expected) +
                      " | Actual: " + std::to_string(actual) + ")")
     << "\n";
  logOutput(ss.str());
  return passed;
}

bool testResult(const std::string &name, const std::vector<float> &actual,
                const std::vector<float> &expected) {
  bool passed = isApproxEqual(actual, expected);
  std::ostringstream ss;
  ss << std::left << std::setw(40) << name + ":";
  ss << (passed ? "PASSED"
                : "FAILED (Expected: " + vecToString(expected) +
                      " | Actual: " + vecToString(actual) + ")")
     << "\n";
  logOutput(ss.str());
  return passed;
}

} // namespace

namespace vectorsearch {

bool testDotProduct() {
  logOutput("\n[Testing dot product]\n");
  return testResult(
             "Basic dot product",
             VectorOps::dotProduct({1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}),
             32.0f) &
         testResult(
             "Zero vector dot product",
             VectorOps::dotProduct({1.0f, 2.0f, 3.0f}, {0.0f, 0.0f, 0.0f}),
             0.0f) &
         testResult(
             "Negative vector dot product",
             VectorOps::dotProduct({-1.0f, -2.0f, -3.0f}, {4.0f, 5.0f, 6.0f}),
             -32.0f);
}

bool testEuclideanDistance() {
  logOutput("\n[Testing euclidean distance]\n");
  return testResult("Basic distance",
                    VectorOps::euclideanDistance({1.0f, 2.0f, 3.0f},
                                                 {4.0f, 5.0f, 6.0f}),
                    std::sqrt(27.0f)) &
         testResult("Same vector distance",
                    VectorOps::euclideanDistance({1.0f, 2.0f, 3.0f},
                                                 {1.0f, 2.0f, 3.0f}),
                    0.0f) &
         testResult("Negative vector distance",
                    VectorOps::euclideanDistance({-1.0f, -2.0f, -3.0f},
                                                 {4.0f, 5.0f, 6.0f}),
                    std::sqrt(155.0f));
}

bool testCosineSimilarity() {
  logOutput("\n[Testing cosine similarity]\n");
  return testResult("Basic similarity",
                    VectorOps::cosineSimilarity({1.0f, 2.0f, 3.0f},
                                                {4.0f, 5.0f, 6.0f}),
                    32.0f / (std::sqrt(14.0f) * std::sqrt(77.0f))) &
         testResult("Same direction similarity",
                    VectorOps::cosineSimilarity({1.0f, 2.0f, 3.0f},
                                                {2.0f, 4.0f, 6.0f}),
                    1.0f) &
         testResult("Orthogonal vectors similarity",
                    VectorOps::cosineSimilarity({1.0f, 2.0f, 3.0f},
                                                {0.0f, 3.0f, -2.0f}),
                    0.0f);
}

bool testNormalize() {
  logOutput("\n[Testing normalization]\n");
  std::vector<float> v1 = {1.0f, 2.0f, 3.0f};
  float norm = std::sqrt(14.0f);
  return testResult("Basic normalization", VectorOps::normalize(v1),
                    {1.0f / norm, 2.0f / norm, 3.0f / norm}) &
         testResult("Zero vector normalization",
                    VectorOps::normalize({0.0f, 0.0f, 0.0f}),
                    {0.0f, 0.0f, 0.0f});
}

} // namespace vectorsearch

int main() {
  logfile.open("vector_ops_tests.log");

  std::time_t now = std::time(nullptr);
  logOutput("Vector Operations Tests - " + std::string(std::ctime(&now)) +
            "\n");

  bool allPassed =
      vectorsearch::testDotProduct() & vectorsearch::testEuclideanDistance() &
      vectorsearch::testCosineSimilarity() & vectorsearch::testNormalize();

  logOutput(allPassed ? "\nAll tests passed!\n" : "\nSome tests failed!\n");
  logfile.close();
  return !allPassed;
}