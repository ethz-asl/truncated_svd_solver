#include "truncated-svd-solver/timing.h"

#include <sys/time.h>

namespace truncated_svd_solver {

Timestamp::Timestamp(double seconds) :
    seconds_(seconds) {
}

Timestamp::Timestamp(const Timestamp& other) :
    seconds_(other.seconds_) {
}

Timestamp& Timestamp::operator = (const Timestamp& other) {
  if (this != &other) {
    seconds_ = other.seconds_;
  }
  return *this;
}

Timestamp::~Timestamp() {}

double Timestamp::getSeconds() const {
  return seconds_;
}

double Timestamp::now() {
  struct timeval time;
  gettimeofday(&time, 0);
  return time.tv_sec + time.tv_usec / 1e6;
}

std::string Timestamp::getDate() {
  struct timeval time;
  gettimeofday(&time, 0);
  struct tm* ptm;
  ptm = localtime(&time.tv_sec);
  char timeString[40];
  strftime(timeString, sizeof (timeString), "%Y-%m-%d %H:%M:%S", ptm);
  return std::string(timeString);
}

std::string Timestamp::getDate(double seconds) {
  struct timeval time;
  time.tv_sec = seconds;
  struct tm* ptm;
  ptm = localtime(&time.tv_sec);
  char timeString[40];
  strftime(timeString, sizeof (timeString), "%Y-%m-%d-%H-%M-%S", ptm);
  return std::string(timeString);
}

Timestamp::operator double() const {
  return seconds_;
}

Timestamp::operator timespec() const {
  timespec time;
  time.tv_sec = (time_t)seconds_;
  time.tv_nsec = (seconds_ - (time_t)seconds_) * 1e9;
  return time;
}

bool Timestamp::operator == (const Timestamp& timestamp) const {
  return (seconds_ == timestamp.seconds_);
}

bool Timestamp::operator == (double seconds) const {
  return (seconds_ == seconds);
}

bool Timestamp::operator != (const Timestamp& timestamp) const {
  return (seconds_ != timestamp.seconds_);
}

bool Timestamp::operator != (double seconds) const {
  return (seconds_ != seconds);
}

bool Timestamp::operator > (const Timestamp& timestamp) const {
  return (seconds_ > timestamp.seconds_);
}

bool Timestamp::operator > (double seconds) const {
  return (seconds_ > seconds);
}

bool Timestamp::operator < (const Timestamp& timestamp) const {
  return (seconds_ < timestamp.seconds_);
}

bool Timestamp::operator < (double seconds) const {
  return (seconds_ < seconds);
}

bool Timestamp::operator >= (const Timestamp& timestamp) const {
  return (seconds_ >= timestamp.seconds_);
}

bool Timestamp::operator >= (double seconds) const {
  return (seconds_ >= seconds);
}

bool Timestamp::operator <= (const Timestamp& timestamp) const {
  return (seconds_ <= timestamp.seconds_);
}

bool Timestamp::operator <= (double seconds) const {
  return (seconds_ <= seconds);
}

Timestamp& Timestamp::operator += (double seconds) {
  seconds_ += seconds;
  return *this;
}

Timestamp& Timestamp::operator -= (double seconds) {
  seconds_ -= seconds;
  return *this;
}

double Timestamp::operator + (double seconds) const {
  return seconds_ + seconds;
}

double Timestamp::operator + (const Timestamp& timestamp) const {
  return seconds_ + timestamp.seconds_;
}

double Timestamp::operator - (const Timestamp& timestamp) const {
  return seconds_ - timestamp.seconds_;
}

double Timestamp::operator - (double seconds) const {
  return seconds_ - seconds;
}

}  // namespace truncated_svd_solver
