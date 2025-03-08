// src/ann/vector_ops.cpp
#include "vector_ops.h"
#include <cmath>
#include <stdexcept>

namespace {

bool isSameDimension(const std::vector<float> &v1,
                     const std::vector<float> &v2) {
  return v1.size() == v2.size();
}

void checkSameDimension(const std::vector<float> &v1,
                        const std::vector<float> &v2) {
  if (!isSameDimension(v1, v2)) {
    throw std::invalid_argument("Vectors must have the same dimension");
  }
}

} // anonymous namespace

namespace vectorsearch {

float VectorOps::dotProduct(const std::vector<float> &v1,
                            const std::vector<float> &v2) {
  checkSameDimension(v1, v2);

  float result = 0.0f;
  for (size_t i = 0; i < v1.size(); ++i) {
    result += v1[i] * v2[i];
  }

  return result;
}

float VectorOps::euclideanDistance(const std::vector<float> &v1,
                                   const std::vector<float> &v2) {
  checkSameDimension(v1, v2);

  float sum = 0.0f;
  for (size_t i = 0; i < v1.size(); ++i) {
    float diff = v1[i] - v2[i];
    sum += diff * diff;
  }

  return std::sqrt(sum);
}

float VectorOps::cosineSimilarity(const std::vector<float> &v1,
                                  const std::vector<float> &v2) {
  checkSameDimension(v1, v2);

  float dot = dotProduct(v1, v2);
  float norm1 = 0.0f;
  float norm2 = 0.0f;

  for (size_t i = 0; i < v1.size(); ++i) {
    norm1 += v1[i] * v1[i];
    norm2 += v2[i] * v2[i];
  }

  if (norm1 == 0.0f || norm2 == 0.0f) { // Avoid division by zero
    return 0.0f;
  }

  norm1 = std::sqrt(norm1);
  norm2 = std::sqrt(norm2);

  return dot / (norm1 * norm2);
}

std::vector<float> VectorOps::normalize(const std::vector<float> &v) {
  float norm = 0.0f;
  for (float val : v) {
    norm += val * val;
  }
  norm = std::sqrt(norm);

  if (norm == 0.0f) { // Avoid division by zero
    return std::vector<float>(v.size(), 0.0f);
  }

  std::vector<float> result(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    result[i] = v[i] / norm;
  }

  return result;
}

} // namespace vectorsearch