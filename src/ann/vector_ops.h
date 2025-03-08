// src/ann/vector_ops.h
#pragma once

#include <cstddef>
#include <vector>

namespace vectorsearch {

class VectorOps final {
public:
  VectorOps() = delete;

  static float dotProduct(const std::vector<float> &v1,
                          const std::vector<float> &v2);

  static float euclideanDistance(const std::vector<float> &v1,
                                 const std::vector<float> &v2);

  static float cosineSimilarity(const std::vector<float> &v1,
                                const std::vector<float> &v2);

  static std::vector<float> normalize(const std::vector<float> &v);

  static bool isSameDimension(const std::vector<float> &v1,
                              const std::vector<float> &v2);
};

} // namespace vectorsearch