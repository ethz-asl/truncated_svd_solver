#ifndef TRUNCATED_SVD_SOLVER_TSVD_TIMING_H
#define TRUNCATED_SVD_SOLVER_TSVD_TIMING_H

#include <string>

#include <time.h>

namespace truncated_svd_solver {

/** The class Timestamp implements timestamping facilities.
    \brief Timestamping facilities
  */
class Timestamp {
 public:
  /// Constructs timestamp object from parameter
  Timestamp(double seconds = now());
  /// Copy constructor
  Timestamp(const Timestamp& other);
  /// Assignment operator
  Timestamp& operator = (const Timestamp& other);
  /// Destructor
  virtual ~Timestamp();

  /// Access the timestamp's value in seconds
  double getSeconds() const;

  /// Return the time in seconds from the timestamp
  operator double() const;
  /// Return the timespec object from the timestamp
  operator timespec() const;
  /// Equal comparison
  bool operator == (const Timestamp& timestamp) const;
  /// Equal comparison
  bool operator == (double seconds) const;
  /// Not equal comparison
  bool operator != (const Timestamp& timestamp) const;
  /// Not equal comparison
  bool operator != (double seconds) const;
  /// Bigger comparison
  bool operator > (const Timestamp& timestamp) const;
  /// Bigger comparison
  bool operator > (double seconds) const;
  /// Smaller comparison
  bool operator < (const Timestamp& timestamp) const;
  /// Smaller comparison
  bool operator < (double seconds) const;
  /// Bigger or equal comparison
  bool operator >= (const Timestamp& timestamp) const;
  /// Bigger or equal comparison
  bool operator >= (double seconds) const;
  /// Smaller or equal comparison
  bool operator <= (const Timestamp& timestamp) const;
  /// Smaller or equal comparison
  bool operator <= (double seconds) const;
  /// Add 2 timestamps
  Timestamp& operator += (double seconds);
  /// Substract 2 timestamps
  Timestamp& operator -= (double seconds);
  /// Add seconds to timestamp
  double operator + (double seconds) const;
  /// Add another timestamp
  double operator + (const Timestamp& timestamp) const;
  /// Substract another timestamp
  double operator - (const Timestamp& timestamp) const;
  /// Substract seconds
  double operator - (double seconds) const;
  /// Returns the system time in s
  static double now();
  /// Returns the date of the system in string
  static std::string getDate();
  /// Returns the date from timestamp in seconds
  static std::string getDate(double seconds);


 private:
  /// Seconds in the timestamp
  double seconds_;
};

}  // namespace truncated_svd_solver

#endif // TRUNCATED_SVD_SOLVER_TSVD_TIMING_H
