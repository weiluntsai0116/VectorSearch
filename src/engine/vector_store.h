// src/engine/vector_store.h
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vectorsearch {

class VectorStore {
public:
  struct VectorRecord {
    std::string id;
    std::vector<float> embedding;
    std::string document_id;
    std::string metadata;
  };

  explicit VectorStore(size_t dimension);

  ~VectorStore();

  bool addVector(const std::string &id, const std::vector<float> &embedding,
                 const std::string &document_id = "",
                 const std::string &metadata = "");

  bool updateVector(const std::string &id, const std::vector<float> &embedding,
                    const std::string &document_id = "",
                    const std::string &metadata = "");

  std::shared_ptr<VectorRecord> getVector(const std::string &id) const;

  bool deleteVector(const std::string &id);

  std::vector<std::shared_ptr<VectorRecord>> getAllVectors() const;

  size_t size() const;

  size_t getDimension() const;

  void clear();

private:
  size_t dimension_;

  std::unordered_map<std::string, std::shared_ptr<VectorRecord>> vectors_;

  mutable std::mutex mutex_;
};

} // namespace vectorsearch