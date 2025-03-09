// test/vector_ops_tests.cpp
#include "ann/vector_ops.h"
#include "test_utils.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

std::ofstream test_utils::logfile;

using namespace test_utils;

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