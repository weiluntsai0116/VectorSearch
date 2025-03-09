// test/test_utils.h
#pragma once

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace test_utils {

extern std::ofstream logfile;

constexpr auto DEFAULT_WIDTH = 50;

// Comparison utilities
inline bool isApproxEqual(float a, float b, float epsilon = 0.0001f) {
  return std::fabs(a - b) < epsilon;
}

inline bool isApproxEqual(const std::vector<float> &v1,
                          const std::vector<float> &v2,
                          float epsilon = 0.0001f) {
  if (v1.size() != v2.size())
    return false;
  for (size_t i = 0; i < v1.size(); ++i) {
    if (!isApproxEqual(v1[i], v2[i], epsilon))
      return false;
  }
  return true;
}

// String conversion utilities
inline std::string vecToString(const std::vector<float> &v) {
  std::ostringstream ss;
  ss << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    ss << v[i] << (i < v.size() - 1 ? ", " : "");
  }
  ss << "]";
  return ss.str();
}

// Logging utilities
inline void logOutput(const std::string &message) {
  std::cout << message;
  if (logfile.is_open()) {
    logfile << message;
  }
}

// Test result reporting
inline bool testResult(const std::string &name, float actual, float expected) {
  bool passed = isApproxEqual(actual, expected);
  std::ostringstream ss;
  ss << std::left << std::setw(DEFAULT_WIDTH) << name + ":";
  ss << (passed ? "PASSED"
                : "FAILED (Expected: " + std::to_string(expected) +
                      " | Actual: " + std::to_string(actual) + ")")
     << "\n";
  logOutput(ss.str());
  return passed;
}

inline bool testResult(const std::string &name, bool actual, bool expected) {
  bool passed = (actual == expected);
  std::ostringstream ss;
  ss << std::left << std::setw(DEFAULT_WIDTH) << name + ":";
  ss << (passed
             ? "PASSED"
             : "FAILED (Expected: " + std::string(expected ? "true" : "false") +
                   " | Actual: " + std::string(actual ? "true" : "false") + ")")
     << "\n";
  logOutput(ss.str());
  return passed;
}

inline bool testResult(const std::string &name, const std::string &actual,
                       const std::string &expected) {
  bool passed = (actual == expected);
  std::ostringstream ss;
  ss << std::left << std::setw(DEFAULT_WIDTH) << name + ":";
  ss << (passed
             ? "PASSED"
             : "FAILED (Expected: " + expected + " | Actual: " + actual + ")")
     << "\n";
  logOutput(ss.str());
  return passed;
}

inline bool testResult(const std::string &name,
                       const std::vector<float> &actual,
                       const std::vector<float> &expected) {
  bool passed = isApproxEqual(actual, expected);
  std::ostringstream ss;
  ss << std::left << std::setw(DEFAULT_WIDTH) << name + ":";
  if (passed) {
    ss << "PASSED";
  } else {
    ss << "FAILED (Expected: " + vecToString(expected) +
              " | Actual: " + vecToString(actual) + ")";
  }
  ss << "\n";
  logOutput(ss.str());
  return passed;
}

template <typename T>
inline bool testResult(const std::string &name, T actual, T expected) {
  bool passed = (actual == expected);
  std::ostringstream ss;
  ss << std::left << std::setw(DEFAULT_WIDTH) << name + ":";
  ss << (passed ? "PASSED"
                : "FAILED (Expected: " + std::to_string(expected) +
                      " | Actual: " + std::to_string(actual) + ")")
     << "\n";
  logOutput(ss.str());
  return passed;
}

} // namespace test_utils