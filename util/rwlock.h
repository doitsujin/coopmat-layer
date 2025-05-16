#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace util {

/**
 * \brief Simple atomic read-write lock
 *
 * Serves as a drop-in replacement for shared_mutex.
 */
class RwLock {
  static constexpr uint32_t ReadBit  = 1u;
  static constexpr uint32_t WriteBit = 1u << 31u;
public:

  RwLock() = default;

  RwLock(const RwLock&) = delete;

  RwLock& operator = (const RwLock&) = delete;

  void lock() {
    auto value = m_lock.load(std::memory_order_relaxed);

    while (value || !m_lock.compare_exchange_strong(value, WriteBit, std::memory_order_acquire, std::memory_order_relaxed)) {
      m_lock.wait(value, std::memory_order_acquire);
      value = m_lock.load(std::memory_order_relaxed);
    }
  }

  bool try_lock() {
    auto value = m_lock.load(std::memory_order_relaxed);

    if (value)
      return false;

    return m_lock.compare_exchange_strong(value, WriteBit, std::memory_order_acquire, std::memory_order_relaxed);
  }

  void unlock() {
    m_lock.store(0u, std::memory_order_release);
    m_lock.notify_all();
  }

  void lock_shared() {
    auto value = m_lock.load(std::memory_order_relaxed);

    do {
      while (value & WriteBit) {
        m_lock.wait(value, std::memory_order_acquire);
        value = m_lock.load(std::memory_order_relaxed);
      }
    } while (!m_lock.compare_exchange_strong(value, value + ReadBit, std::memory_order_acquire, std::memory_order_relaxed));
  }

  bool try_lock_shared() {
    auto value = m_lock.load(std::memory_order_relaxed);

    if (value & WriteBit)
      return false;

    return m_lock.compare_exchange_strong(value, value + ReadBit, std::memory_order_acquire, std::memory_order_relaxed);
  }

  void unlock_shared() {
    m_lock.fetch_sub(ReadBit, std::memory_order_release);
    m_lock.notify_one();
  }

private:

  std::atomic<uint32_t> m_lock = { 0u };

};

};
