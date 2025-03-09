// test/vector_store_tests.cpp
#include "engine/vector_store.h"
#include "test_utils.h"
#include <ctime>
#include <thread>

std::ofstream test_utils::logfile;

using namespace test_utils;

namespace vectorsearch {

bool testBasicOperations() {
  logOutput("\n[Testing basic vector store operations]\n");

  // Create a vector store
  const size_t dimension = 3;
  VectorStore store(dimension);

  // Test initial state
  bool passed =
      testResult("Initial size", store.size(), static_cast<size_t>(0));
  passed &= testResult("Dimension", store.getDimension(), dimension);

  // Add a vector
  const std::string id = "vec1";
  const std::vector<float> embedding = {1.0f, 2.0f, 3.0f};
  const std::string docId = "doc1";
  const std::string metadata = R"({"key": "value"})";

  passed &= testResult("Add vector",
                       store.addVector(id, embedding, docId, metadata), true);
  passed &= testResult("Size after add", store.size(), static_cast<size_t>(1));

  // Get the vector back
  auto vector = store.getVector(id);
  passed &= testResult("Get vector returns non-null", vector != nullptr, true);

  if (vector) {
    passed &= testResult("Vector ID", vector->id, id);
    passed &= testResult("Vector embedding", vector->embedding, embedding);
    passed &= testResult("Vector document ID", vector->document_id, docId);
    passed &= testResult("Vector metadata", vector->metadata, metadata);
  }

  // Update the vector
  const std::vector<float> newEmbedding = {4.0f, 5.0f, 6.0f};
  const std::string newMetadata = R"({"key": "new_value"})";

  passed &=
      testResult("Update vector",
                 store.updateVector(id, newEmbedding, "", newMetadata), true);

  // Get the updated vector
  vector = store.getVector(id);
  passed &= testResult("Get updated vector returns non-null", vector != nullptr,
                       true);

  if (vector) {
    passed &= testResult("Updated vector ID", vector->id, id);
    passed &=
        testResult("Updated vector embedding", vector->embedding, newEmbedding);
    passed &= testResult("Preserved original document ID", vector->document_id,
                         docId);
    passed &=
        testResult("Updated vector metadata", vector->metadata, newMetadata);
  }

  // Test selective updates
  logOutput("\n[Testing selective updates]\n");

  // Update only metadata
  const std::string newerMetadata = R"({"key": "newer_value"})";
  passed &= testResult("Update only metadata",
                       store.updateVector(id, {}, "", newerMetadata), true);

  vector = store.getVector(id);
  if (vector) {
    passed &= testResult("Preserved embedding during metadata-only update",
                         vector->embedding, newEmbedding);
    passed &= testResult("Updated metadata in selective update",
                         vector->metadata, newerMetadata);
  }

  // Update only embedding
  const std::vector<float> newestEmbedding = {7.0f, 8.0f, 9.0f};
  passed &= testResult("Update only embedding",
                       store.updateVector(id, newestEmbedding), true);

  vector = store.getVector(id);
  if (vector) {
    passed &= testResult("Updated embedding in selective update",
                         vector->embedding, newestEmbedding);
    passed &= testResult("Preserved metadata during embedding-only update",
                         vector->metadata, newerMetadata);
  }

  // Delete the vector
  passed &= testResult("Delete vector", store.deleteVector(id), true);
  passed &=
      testResult("Size after delete", store.size(), static_cast<size_t>(0));

  // Try to get a non-existent vector
  vector = store.getVector(id);
  passed &= testResult("Get non-existent vector returns null",
                       vector == nullptr, true);

  return passed;
}

bool testMultipleVectors() {
  logOutput("\n[Testing multiple vectors in store]\n");

  // Create a vector store
  const size_t dimension = 3;
  VectorStore store(dimension);

  // Add multiple vectors
  bool passed = true;
  const size_t numVectors = 5;

  for (size_t i = 0; i < numVectors; ++i) {
    std::string id = "vec" + std::to_string(i);
    std::vector<float> embedding = {static_cast<float>(i),
                                    static_cast<float>(i + 1),
                                    static_cast<float>(i + 2)};
    std::string docId = "doc" + std::to_string(i);
    std::string metadata = "{\"index\": " + std::to_string(i) + "}";

    passed &= testResult("Add vector " + id,
                         store.addVector(id, embedding, docId, metadata), true);
  }

  passed &= testResult("Size after adding multiple", store.size(), numVectors);

  // Get all vectors
  auto allVectors = store.getAllVectors();
  passed &= testResult("Get all vectors size", allVectors.size(), numVectors);

  // Clear the store
  store.clear();
  passed &=
      testResult("Size after clear", store.size(), static_cast<size_t>(0));

  return passed;
}

bool testDimensionCheck() {
  logOutput("\n[Testing dimension validation]\n");

  // Create a vector store
  const size_t dimension = 3;
  VectorStore store(dimension);

  // Try to add a vector with incorrect dimension
  const std::string id = "vec1";
  const std::vector<float> wrongDimEmbedding = {1.0f, 2.0f};
  bool exceptionThrown = false;

  try {
    store.addVector(id, wrongDimEmbedding);
  } catch (const std::invalid_argument &e) {
    exceptionThrown = true;
  }

  return testResult("Exception on wrong dimension", exceptionThrown, true);
}

bool testThreadSafety() {
  logOutput("\n[Testing thread safety]\n");

  // Create a vector store
  const size_t dimension = 3;
  VectorStore store(dimension);

  // Add vectors from multiple threads
  const size_t numThreads = 10;
  const size_t vectorsPerThread = 100;
  std::vector<std::thread> threads;

  for (size_t t = 0; t < numThreads; ++t) {
    threads.emplace_back([&store, t, dimension, vectorsPerThread]() {
      for (size_t i = 0; i < vectorsPerThread; ++i) {
        std::string id =
            "thread" + std::to_string(t) + "_vec" + std::to_string(i);
        std::vector<float> embedding(
            dimension, static_cast<float>(t * vectorsPerThread + i));
        store.addVector(id, embedding);
      }
    });
  }

  // Wait for all threads to complete
  for (auto &thread : threads) {
    thread.join();
  }

  // Check if all vectors were added correctly
  const size_t expectedSize = numThreads * vectorsPerThread;
  bool passed =
      testResult("Size after multithreaded adds", store.size(), expectedSize);

  return passed;
}

} // namespace vectorsearch

int main() {
  logfile.open("vector_store_tests.log");

  std::time_t now = std::time(nullptr);
  logOutput("Vector Store Tests - " + std::string(std::ctime(&now)) + "\n");

  bool allPassed = vectorsearch::testBasicOperations() &
                   vectorsearch::testMultipleVectors() &
                   vectorsearch::testDimensionCheck() &
                   vectorsearch::testThreadSafety();

  logOutput(allPassed ? "\nAll tests passed!\n" : "\nSome tests failed!\n");
  logfile.close();
  return !allPassed;
}