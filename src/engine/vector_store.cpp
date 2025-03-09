// src/engine/vector_store.cpp
#include "vector_store.h"
#include <stdexcept>

namespace vectorsearch {

VectorStore::VectorStore(size_t dimension) : dimension_(dimension) {}

VectorStore::~VectorStore() = default;

bool VectorStore::addVector(const std::string &id,
                            const std::vector<float> &embedding,
                            const std::string &document_id,
                            const std::string &metadata) {
  // Check if the embedding has the correct dimension
  if (embedding.size() != dimension_) {
    throw std::invalid_argument(
        "Embedding dimension (" + std::to_string(embedding.size()) +
        ") doesn't match store dimension (" + std::to_string(dimension_) + ")");
  }

  std::lock_guard<std::mutex> lock(mutex_);

  // Check if the ID already exists
  if (vectors_.find(id) != vectors_.end()) {
    return false;
  }

  // Create the vector record
  auto record = std::make_shared<VectorRecord>();
  record->id = id;
  record->embedding = embedding;
  record->document_id = document_id;
  record->metadata = metadata;

  // Store the vector
  vectors_[id] = record;
  return true;
}

bool VectorStore::updateVector(const std::string &id,
                               const std::vector<float> &embedding,
                               const std::string &document_id,
                               const std::string &metadata) {
  // Check if the embedding has the correct dimension
  if (!embedding.empty() && (embedding.size() != dimension_)) {
    throw std::invalid_argument(
        "Embedding dimension (" + std::to_string(embedding.size()) +
        ") doesn't match store dimension (" + std::to_string(dimension_) + ")");
  }

  std::lock_guard<std::mutex> lock(mutex_);

  // Check if the ID exists
  auto it = vectors_.find(id);
  if (it == vectors_.end()) {
    return false;
  }

  // Update the vector record
  if (!embedding.empty()) {
    it->second->embedding = embedding;
  }
  if (!document_id.empty()) {
    it->second->document_id = document_id;
  }
  if (!metadata.empty()) {
    it->second->metadata = metadata;
  }

  return true;
}

std::shared_ptr<VectorStore::VectorRecord>
VectorStore::getVector(const std::string &id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = vectors_.find(id);
  if (it == vectors_.end()) {
    return nullptr;
  }

  return it->second;
}

bool VectorStore::deleteVector(const std::string &id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = vectors_.find(id);
  if (it == vectors_.end()) {
    return false;
  }

  vectors_.erase(it);
  return true;
}

std::vector<std::shared_ptr<VectorStore::VectorRecord>>
VectorStore::getAllVectors() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<std::shared_ptr<VectorRecord>> result;
  result.reserve(vectors_.size());

  for (const auto &pair : vectors_) {
    result.emplace_back(pair.second);
  }

  return result;
}

size_t VectorStore::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return vectors_.size();
}

size_t VectorStore::getDimension() const { return dimension_; }

void VectorStore::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  vectors_.clear();
}

} // namespace vectorsearch